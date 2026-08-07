# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
from torch import nn
from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

logger = init_logger(__name__)


class SequentialOffloadHook(ModelHook):
    """Hook for sequential offloading with mutual exclusion on encoder and DiT modules.

    To be used as a model-level (or "component-level") of CPU offloading method;
    When a module's forward is called, this hook offloads target modules to CPU
    and loads the current module to GPU.
    """

    _HOOK_NAME = "sequential_offload"

    def __init__(
        self,
        offload_targets: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ):
        # Modules to offload to CPU before this module runs
        self.offload_targets = offload_targets
        self.device = device
        self.pin_memory = pin_memory
        self.use_hsdp = use_hsdp

    @staticmethod
    def _move_params(
        module: nn.Module,
        target_device: torch.device,
        *,
        non_blocking: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """Move module parameters and buffers to device.

        This cls method specifically prevents recursion device movement,
        E.g., Cache-DiT CachedBlocks has attr `transformer` as a ref to original
        transformer blocks, thus `module.to(device)` will fail for recursion calling,
        refer to
        https://github.com/vipshop/cache-dit/blob/v1.2.3/src/cache_dit/caching/cache_blocks/__init__.py#L83
        """
        for p in module.parameters():
            if p.data.device != target_device:
                data = p.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                p.data = data
        for b in module.buffers():
            if b.device != target_device:
                data = b.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                b.data = data

    def _to_cpu(self, module: nn.Module) -> None:
        try:
            param = next(module.parameters())
        except StopIteration:
            return

        if param.device.type == "cpu":
            return

        # XPU's allocator doesn't respect stream dependencies in empty_cache,
        # so non-blocking copies can race with cache eviction. Use blocking
        # copies on XPU to avoid NULL pointer errors during DMA.
        non_blocking = not self.use_hsdp and not current_omni_platform.is_xpu()
        self._move_params(
            module,
            torch.device("cpu"),
            non_blocking=non_blocking,
            pin_memory=self.pin_memory,
        )
        # empty_cache() hands cached blocks back to the driver. Per the note
        # above, the XPU allocator does not honour stream dependencies here,
        # so any buffer still being written by an in-flight kernel -- the DiT
        # latents this hook is about to hand to the VAE, above all -- can be
        # released and reissued underneath that kernel. pre_forward() only
        # synchronizes after the whole evict-then-load sequence, which is too
        # late. EC_MODE selects the arm; see patch_empty_cache_order.py.
        _ec_mode = os.environ.get("EC_MODE", "upstream").strip().lower()
        if _ec_mode == "sync":
            current_omni_platform.synchronize()
            current_omni_platform.empty_cache()
        elif _ec_mode == "none":
            pass
        else:
            current_omni_platform.empty_cache()

    def _to_gpu(self, module: nn.Module) -> None:
        try:
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        self._move_params(module, self.device, non_blocking=False)

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        # Offload target modules to CPU
        for target in self.offload_targets:
            self._to_cpu(target)

        # Load current module to GPU
        self._to_gpu(module)
        current_omni_platform.synchronize()

        logger.debug(
            "Swapped: %s -> CPU, %s -> %s, free memory: %.4f GB",
            [t.__class__.__name__ for t in self.offload_targets],
            module.__class__.__name__,
            f"{self.device.type}:{self.device.index}",
            current_omni_platform.get_free_memory() / 1024 / 1024 / 1024,
        )

        return args, kwargs


def apply_sequential_offload(
    dit_modules: list[nn.Module],
    encoder_modules: list[nn.Module],
    device: torch.device,
    pin_memory: bool = True,
    use_hsdp: bool = False,
    vae_modules: list[nn.Module] | None = None,
) -> None:
    """Apply sequential offloading hooks to DiT and encoder modules.

    Registers hooks on modules to implement mutual-exclusion GPU allocation.
        - Before DiT runs, encoders are offloaded to CPU.
        - Before encoders run, DiT is offloaded to CPU.

    Args:
        dit_modules: DiT/transformer modules to register hooks on
        encoder_modules: Encoder modules to register hooks on
        device: Target GPU device for loading
        pin_memory: Whether to pin CPU memory for faster transfers
        use_hsdp: Whether HSDP is enabled (affects non_blocking behavior)

    Example:
        >>> apply_sequential_offload(
        ...     dit_modules=[pipeline.transformer],
        ...     encoder_modules=[pipeline.text_encoder, pipeline.vae],
        ...     device=torch.device("cuda:0"),
        ... )
        >>> # Modules of pipeline now automatically swap between CPU and GPU
    """
    vaes = list(vae_modules or [])

    # Register hooks on DiT modules (offload encoders AND other DiTs when a DiT runs)
    for i, dit_mod in enumerate(dit_modules):
        other_dits = [d for j, d in enumerate(dit_modules) if j != i]
        registry = HookRegistry.get_or_create(dit_mod)
        hook = SequentialOffloadHook(
            offload_targets=encoder_modules + other_dits + vaes,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for %s", dit_mod.__class__.__name__)

    # Register hooks on encoders (offload DiTs when encoder runs)
    for enc in encoder_modules:
        registry = HookRegistry.get_or_create(enc)
        hook = SequentialOffloadHook(
            offload_targets=dit_modules + vaes,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for %s", enc.__class__.__name__)

    # Register hooks on VAEs (offload DiTs and encoders when a VAE runs).
    # Only populated when vae_cpu_offload is set; otherwise VAEs stay
    # device-resident as SupportsComponentDiscovery documents.
    for vae in vaes:
        registry = HookRegistry.get_or_create(vae)
        hook = SequentialOffloadHook(
            offload_targets=dit_modules + encoder_modules,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug("Registered offload hook for VAE %s", vae.__class__.__name__)


def remove_sequential_offload(modules: list[nn.Module]) -> None:
    """Remove sequential offloading hooks from modules.

    Args:
        modules: Modules to remove hooks from

    Example:
        >>> all_modules = [*dit_modules, *encoder_modules]
        >>> remove_sequential_offload(all_modules)
    """
    for module in modules:
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        if registry is not None:
            registry.remove_hook(SequentialOffloadHook._HOOK_NAME)
            logger.debug("Removed offload hook from %s", module.__class__.__name__)


class ModelLevelOffloadBackend(OffloadBackend):
    """Model-level (sequential) offloading backend.

    Uses SequentialOffloadHook registered via HookRegistry for automatic module swapping.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)
        self._offload_modules: list[nn.Module] = []  # Track modules with hooks
        self._custom_pipeline: nn.Module | None = None

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("ModelLevelOffloadBackend already enabled")
            return

        # Pipelines with a nested transformer (e.g. Cosmos3's reasoner/generator
        # pathways) own their own mutual-exclusion swaps and cannot be offloaded
        # with the generic encoder<->DiT hooks. Delegate to the custom hook.
        # TODO: Cosmos3 is the only implementer today, so we duck-type the
        # enable/disable_omni_model_cpu_offload pair via getattr. Once a second
        # pipeline needs it, formalize this as a Protocol (like
        # SupportsComponentDiscovery) instead of getattr detection.
        custom_enable = getattr(pipeline, "enable_omni_model_cpu_offload", None)
        if callable(custom_enable):
            custom_enable(
                device=self.device,
                pin_memory=self.config.pin_cpu_memory,
                use_hsdp=self.config.use_hsdp,
            )
            self._custom_pipeline = pipeline
            self.enabled = True
            logger.info(
                "Model-level offloading enabled through %s.enable_omni_model_cpu_offload",
                pipeline.__class__.__name__,
            )
            return

        modules = ModuleDiscovery.discover(pipeline)

        # Move encoders to GPU
        for enc in modules.encoders:
            enc.to(self.device)

        # VAEs are device-resident by default. With vae_cpu_offload they are
        # parked on the host instead -- note the explicit move to CPU: a
        # pipeline may already have built its VAE on the device (MiniMax-H3
        # does, in FP32), so not-moving-to-GPU is not enough to free it.
        offload_vaes = bool(getattr(self.config, "vae_cpu_offload", False))
        for vae in modules.vaes:
            try:
                vae.to(torch.device("cpu") if offload_vaes else self.device, non_blocking=not offload_vaes)
            except Exception as exc:
                # A swallowed failure here is indistinguishable from a successful
                # offload: both leave the VAE where it was and print nothing at
                # INFO. Warn, so the next reader is not left measuring memory to
                # find out whether the move happened.
                logger.warning("Failed to place VAE: %s", exc)

        if offload_vaes:
            # Moving a module off the device returns its blocks to the caching
            # allocator, not to the driver -- so xpu-smi keeps reporting the old
            # figure and the offload looks like a no-op. Release the cache so the
            # freed VAE memory is actually available to the DiT.
            empty_cache = getattr(getattr(torch, self.device.type, None), "empty_cache", None)
            if empty_cache is not None:
                empty_cache()

        # Pin resident modules on GPU (small hot submodules called inside the DiT loop).
        for res, name in zip(modules.resident_modules, modules.resident_names):
            try:
                res.to(self.device)
            except Exception as exc:
                logger.warning("Failed to move resident module '%s' to GPU: %s", name, exc)

        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping model-level offloading")
            return

        if not modules.encoders:
            # Nothing to swap against — move DiTs to GPU and skip hooks.
            for dit in modules.dits:
                dit.to(self.device)
            logger.warning("No encoder modules found, skipping model-level offloading")
            return

        # Apply sequential offloading hooks
        apply_sequential_offload(
            dit_modules=modules.dits,
            encoder_modules=modules.encoders,
            device=self.device,
            pin_memory=self.config.pin_cpu_memory,
            use_hsdp=self.config.use_hsdp,
            vae_modules=modules.vaes if offload_vaes else None,
        )

        # Track modules for cleanup
        self._offload_modules = [*modules.dits, *modules.encoders]
        if offload_vaes:
            self._offload_modules.extend(modules.vaes)

        self.enabled = True

        logger.info(
            "Model-level offloading enabled: %s <-> %s (mutual exclusion)%s",
            ", ".join(modules.dit_names),
            ", ".join(modules.encoder_names),
            f"; resident on GPU: {', '.join(modules.resident_names)}" if modules.resident_names else "",
        )

    def disable(self) -> None:
        if not self.enabled:
            return

        if self._custom_pipeline is not None:
            custom_disable = getattr(self._custom_pipeline, "disable_omni_model_cpu_offload", None)
            if callable(custom_disable):
                custom_disable()
            self._custom_pipeline = None
            self.enabled = False
            logger.info("Model-level offloading disabled")
            return

        remove_sequential_offload(self._offload_modules)

        self._offload_modules.clear()
        self.enabled = False
        logger.info("Model-level offloading disabled")
