# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit component residency for transient (on-demand) modules.

Motivation
----------
Layer-wise offloading keeps one component (e.g. the text encoder) streaming its
blocks from the host while every other component stays whole on the device. On
MiniMax-H3 a single card is effectively full during denoising (32.6 GiB used of
a 32.65 GiB card), so a VAE that has been parked on the host cannot simply be
pulled in: the whole-resident DiT has to make room for it first.

The obvious way to express that is a *hook net*: give every component a
pre-forward hook that loads itself and evicts the others, so the restore of an
evicted component happens lazily inside the next component's own pre_forward.
That is what model-level (sequential) offloading does, and it is why that
strategy is self-consistent. It was tried here and rejected on evidence:

* Hooking only the VAE is one-directional -- the DiT is evicted and nobody ever
  brings it back, so the next denoise step dies on a device mismatch (H3's RoPE
  ``inv_freq`` buffer is the first tensor to notice).
* Adding the counter-hook on the DiT closes the loop but is not stable. The
  eviction path calls ``empty_cache()`` *without* synchronizing first, and the
  XPU allocator does not honour stream dependencies there (see the note in
  ``sequential_backend.SequentialOffloadHook._to_cpu``), so the swap races with
  in-flight kernels. Observed as ``level_zero backend failed with error: 40`` on
  the first DiT forward after a VAE decode, and as a worker wedged during init.

This module takes the approach sglang uses instead (``memory_managers/
component_resident_strategies.py``): residency is *declared explicitly* around
a stage boundary via acquire/release, transfers are whole-module (parameters
**and** buffers, which is what the ``inv_freq`` failure was about), and every
device-to-host move is synchronized before the cache is released so no transfer
overlaps a live kernel.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import ModelHook
from vllm_omni.platforms import current_omni_platform

from .sequential_backend import SequentialOffloadHook

logger = init_logger(__name__)


def module_reference_tensor(module: nn.Module) -> torch.Tensor | None:
    """Return a tensor that stands in for the module's residency state.

    Parameters come first, buffers second. A module whose parameters are on the
    device but whose buffers are not is *not* ready -- that asymmetry is exactly
    the H3 RoPE ``inv_freq`` bug -- but since every move in this file is
    whole-module, checking the first available tensor is sufficient and matches
    ``_module_reference_tensor`` in the sglang implementation.
    """
    for p in module.parameters():
        return p.data
    for b in module.buffers():
        return b.data
    return None


def module_resident_on(module: nn.Module, device: torch.device) -> bool:
    """True if the module already lives on ``device`` (no transfer needed)."""
    ref = module_reference_tensor(module)
    if ref is None:
        return True
    if ref.device.type != device.type:
        return False
    # A device index of None means "current device" and matches any index.
    if device.index is None or ref.device.index is None:
        return True
    return ref.device.index == device.index


class ResidencyCoordinator:
    """Coordinates a set of transient modules against whole-resident ones.

    ``transients`` are parked on the host and pulled in only for the duration of
    an explicit scope (``acquire``/``release``). ``residents`` are the modules
    that have to be evicted to make room, and -- unlike a hook net -- they are
    restored by this object at the end of the scope rather than by somebody
    else's pre_forward.

    Scopes refcount, so a stage that touches two transients back to back (H3's
    ``decode()`` runs the video VAE and then the audio VAE) pays for the
    resident round trip once, not twice.
    """

    def __init__(
        self,
        residents: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ) -> None:
        self.residents = residents
        self.device = device
        self.pin_memory = pin_memory
        self.use_hsdp = use_hsdp
        self._depth = 0
        self._evicted: list[nn.Module] = []
        self._active: list[nn.Module] = []

    # -- transfers -------------------------------------------------------
    def _to_host(self, module: nn.Module) -> None:
        if module_resident_on(module, torch.device("cpu")):
            return
        SequentialOffloadHook._move_params(
            module,
            torch.device("cpu"),
            non_blocking=False,
            pin_memory=self.pin_memory,
        )

    def _to_device(self, module: nn.Module) -> None:
        if module_resident_on(module, self.device):
            return
        SequentialOffloadHook._move_params(module, self.device, non_blocking=False)

    def _release_cache(self) -> None:
        """Hand freed blocks back to the driver, but only once it is safe.

        The synchronize() is load-bearing and is the difference between this
        path and the hook net it replaces: the XPU caching allocator does not
        respect stream dependencies in empty_cache(), so releasing while a
        kernel is still writing a buffer can hand that memory out again
        underneath it.
        """
        current_omni_platform.synchronize()
        current_omni_platform.empty_cache()

    # -- scope -----------------------------------------------------------
    def acquire(self, module: nn.Module) -> None:
        """Make ``module`` resident, evicting the whole-resident set if needed."""
        if self._depth == 0:
            self._evicted = []
            for resident in self.residents:
                if module_resident_on(resident, self.device):
                    self._to_host(resident)
                    self._evicted.append(resident)
            if self._evicted:
                self._release_cache()
        self._depth += 1
        self._active.append(module)
        self._to_device(module)
        current_omni_platform.synchronize()
        logger.debug(
            "Residency: acquired %s (depth=%d, evicted=%d, free=%.2f GB)",
            module.__class__.__name__,
            self._depth,
            len(self._evicted),
            current_omni_platform.get_free_memory() / 1024**3,
        )

    def release(self, module: nn.Module) -> None:
        """Park ``module`` back on the host and restore the evicted residents."""
        try:
            self._active.remove(module)
        except ValueError:
            pass
        self._depth = max(0, self._depth - 1)
        if self._depth > 0:
            # Another transient scope is still open; keep the residents out.
            return
        self._to_host(module)
        self._release_cache()
        for resident in self._evicted:
            self._to_device(resident)
        if self._evicted:
            current_omni_platform.synchronize()
        logger.debug(
            "Residency: released %s, restored %d resident(s), free=%.2f GB",
            module.__class__.__name__,
            len(self._evicted),
            current_omni_platform.get_free_memory() / 1024**3,
        )
        self._evicted = []


class TransientResidencyHook(ModelHook):
    """Wraps a transient module's forward in an explicit residency scope.

    ``new_forward`` (rather than pre/post_forward) is used deliberately: it is
    the only place where the release can be put in a ``finally``. With a plain
    post_forward hook a failed decode would leave the residents on the host and
    poison every subsequent request -- which is precisely the failure mode this
    patch exists to remove.
    """

    _HOOK_NAME = "transient_residency"

    def __init__(self, coordinator: ResidencyCoordinator) -> None:
        self.coordinator = coordinator

    def new_forward(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        self.coordinator.acquire(module)
        try:
            return module._omni_original_forward(*args, **kwargs)  # type: ignore[attr-defined]
        finally:
            self.coordinator.release(module)
