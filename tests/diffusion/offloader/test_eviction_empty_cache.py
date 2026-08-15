# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The per-eviction ``empty_cache`` must not run on XPU.

Returning the freed segments to the driver is not what makes the HBM reusable
-- the caching allocator already recycles those blocks. On XPU it does have a
cost: it churns the addresses of the collective receive buffers allocated
after it, and the XPU collective backend keeps a non-reclaimable driver
registration per distinct receive-buffer address, so device memory outside the
PyTorch pool was observed growing across the measured requests.

These tests pin the platform split: XPU skips the call, every other platform
keeps the previous behaviour byte for byte.
"""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.module_residency import PinnedModuleStager
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture
def platform_probe(monkeypatch):
    """Drive ``is_xpu`` and count ``empty_cache`` without touching a device."""
    calls: list[str] = []

    def _install(*, is_xpu: bool):
        monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: is_xpu, raising=False)
        monkeypatch.setattr(
            current_omni_platform, "empty_cache", lambda: calls.append("empty_cache"), raising=False
        )
        monkeypatch.setattr(
            current_omni_platform, "synchronize", lambda: calls.append("synchronize"), raising=False
        )
        return calls

    return _install


def _hook_on_cpu_target() -> tuple[SequentialOffloadHook, nn.Module]:
    module = nn.Linear(4, 4)
    hook = SequentialOffloadHook([module], torch.device("cpu"), pin_memory=False)
    return hook, module


def _staged_module() -> PinnedModuleStager:
    """A stager in the ``loaded`` state without requiring an accelerator."""
    stager = object.__new__(PinnedModuleStager)
    master = torch.zeros(4)
    target = torch.zeros(4)
    stager.loaded = True
    stager._entries = [(target, master)]
    stager._device_tensors = [target]
    return stager


@pytest.mark.parametrize("is_xpu, expected", [(True, 0), (False, 1)])
def test_stager_offload_skips_empty_cache_only_on_xpu(platform_probe, is_xpu, expected) -> None:
    calls = platform_probe(is_xpu=is_xpu)
    stager = _staged_module()

    stager.offload()

    assert stager.loaded is False
    assert calls.count("synchronize") == 1, "the stage-boundary sync must run on every platform"
    assert calls.count("empty_cache") == expected


@pytest.mark.parametrize("is_xpu, expected", [(True, 0), (False, 1)])
def test_sequential_to_cpu_skips_empty_cache_only_on_xpu(
    platform_probe, monkeypatch, is_xpu, expected
) -> None:
    calls = platform_probe(is_xpu=is_xpu)
    hook, module = _hook_on_cpu_target()
    # ``_to_cpu`` returns early when the module already lives on CPU, which
    # would make the assertion pass for the wrong reason. ``meta`` gives a
    # non-CPU device without needing an accelerator; the transfer itself is
    # stubbed out because meta storage cannot be copied.
    module.to(torch.device("meta"))
    monkeypatch.setattr(
        SequentialOffloadHook,
        "_move_params",
        staticmethod(lambda *args, **kwargs: calls.append("move_params")),
    )

    hook._to_cpu(module)

    assert calls.count("move_params") == 1, "the eviction itself must still happen"
    assert calls.count("empty_cache") == expected
