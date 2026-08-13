# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class OffloadStrategy(Enum):
    NONE = "none"
    MODEL_LEVEL = "model_level"  # Sequential offloading between DiT and encoders
    LAYER_WISE = "layer_wise"  # Block-level
    DISTRIBUTED_LAYER_WISE = "distributed_layer_wise"  # Block-level with DP sharding + H2D/AllGather overlap


@dataclass
class OffloadConfig:
    strategy: OffloadStrategy
    pin_cpu_memory: bool = True
    use_hsdp: bool = False
    dp_size: int = 1  # derived from parallel_config, not user-configurable
    # True: add DP sharding + AllGather. False: stream complete rank-local
    # blocks from the loader-selected host backing with H2D only.
    dlo_use_allgather: bool = True
    dlo_resident_layers: int = 0  # leading DiT layers kept on device
    model_path: str | None = None  # checkpoint path for mmap weight loading
    # Operator policy: under model-level offload, allow pipeline-staged VAEs to
    # stay in host memory instead of resident on the device.
    vae_cpu_offload: bool = False

    @classmethod
    def from_od_config(cls, od_config: OmniDiffusionConfig) -> "OffloadConfig":
        """Extract and validate offload settings from OmniDiffusionConfig.

        Enforces mutual exclusion among the three offload strategies.
        Distributed layer-wise takes the highest priority, then layer-wise,
        then model-level.

        The ``dp_size`` is automatically derived from ``parallel_config`` —
        it is NOT a user-configurable parameter. The distributed layerwise
        offload works with whatever DP/SP parallelism is already set up.

        Args:
            od_config: OmniDiffusionConfig with offload settings

        Returns:
            OffloadConfig with validated settings
        """
        enable_cpu_offload = getattr(od_config, "enable_cpu_offload", False)
        vae_cpu_offload = getattr(od_config, "vae_cpu_offload", False)
        enable_layerwise_offload = getattr(od_config, "enable_layerwise_offload", False)
        enable_distributed_layerwise_offload = getattr(od_config, "enable_distributed_layerwise_offload", False)
        pin_cpu_memory = getattr(od_config, "pin_cpu_memory", True)

        parallel_config = getattr(od_config, "parallel_config", None)
        use_hsdp = getattr(parallel_config, "use_hsdp", False) if parallel_config else False
        # Derive dp_size from parallel_config — not user-configurable.
        # The offload adapts to whatever DP/SP is already configured.
        dp_size = 1
        if parallel_config is not None:
            dp_size = getattr(parallel_config, "data_parallel_size", 1)
            # HSDP's fully_shard_degree also contributes to effective DP
            hsdp_shard_size = getattr(parallel_config, "hsdp_shard_size", -1) if use_hsdp else -1
            hsdp_replicate_size = getattr(parallel_config, "hsdp_replicate_size", 1) if use_hsdp else 1
            if use_hsdp and hsdp_shard_size > 0:
                dp_size = hsdp_shard_size * hsdp_replicate_size

            # When there is no DP but SP > 1, shard weights across SP ranks.
            # AllGather reconstructs full weights per layer; each rank then
            # computes on its SP portion of the sequence.  This gives N×
            # compute parallelism with 1/N H2D transfer, reusing the exact
            # same AllGather code path — only the process group changes.
            if dp_size <= 1:
                sp_size = getattr(parallel_config, "sequence_parallel_size", 1)
                if sp_size and sp_size > 1:
                    dp_size = sp_size

        # Determine strategy (mutual exclusion, distributed layer-wise takes priority)
        if enable_distributed_layerwise_offload:
            strategy = OffloadStrategy.DISTRIBUTED_LAYER_WISE
            if enable_layerwise_offload or enable_cpu_offload:
                logger.info("Distributed layer-wise offloading takes priority, disabling other offloading strategies.")
        elif enable_layerwise_offload:
            strategy = OffloadStrategy.LAYER_WISE
            if enable_cpu_offload:
                logger.info(
                    "Both model-level and layer-wise offloading enabled. "
                    "Layer-wise takes priority, disabling model-level offloading."
                )
        elif enable_cpu_offload:
            strategy = OffloadStrategy.MODEL_LEVEL
        else:
            strategy = OffloadStrategy.NONE

        # With dlo_use_allgather=False, do not add another DP shard. Each rank
        # streams the tensors produced by the standard loader, which may
        # already be TP-local shards. This avoids AllGather synchronization
        # requirements (concurrent requests, dummy run skip).
        dlo_use_allgather = getattr(od_config, "dlo_use_allgather", True)
        dlo_resident_layers = int(getattr(od_config, "dlo_resident_layers", 0))
        if dlo_resident_layers < 0:
            raise ValueError(f"dlo_resident_layers must be >= 0, got {dlo_resident_layers}")
        if dlo_resident_layers and dlo_use_allgather:
            raise ValueError(
                "dlo_resident_layers currently requires --dlo-no-use-allgather so "
                "resident blocks use weights prepared by the standard TP-aware loader"
            )

        # If dlo_use_allgather=False, force dp_size=1 (each rank independent)
        if enable_distributed_layerwise_offload and not dlo_use_allgather:
            dp_size = 1
            logger.info(
                "Distributed layerwise offload: dlo_use_allgather=False, "
                "streaming complete rank-local blocks (no DLO shard or AllGather); "
                "the backend will select mmap or standard-loader host storage"
            )

        # HSDP already shards parameters into DTensors.  Running distributed
        # layerwise offload on top would shard each to_local() again, producing
        # incorrect reconstruction after AllGather.  Reject this combination.
        if enable_distributed_layerwise_offload and use_hsdp and dlo_use_allgather:
            raise ValueError(
                "Distributed layerwise offload with AllGather is incompatible with "
                "HSDP: HSDP parameters are already sharded DTensors, and the offloader "
                "would double-shard them. Use --dlo-no-use-allgather (standard-loader "
                "rank-local weights) or disable HSDP."
            )

        return cls(
            strategy=strategy,
            pin_cpu_memory=pin_cpu_memory,
            use_hsdp=use_hsdp,
            dp_size=dp_size,
            dlo_use_allgather=dlo_use_allgather,
            dlo_resident_layers=dlo_resident_layers,
            model_path=getattr(od_config, "model", None),
            vae_cpu_offload=bool(vae_cpu_offload),
        )


def can_stage_on_demand(module: nn.Module) -> bool:
    """Return whether a component loads and releases itself on demand.

    Pipelines that manage their own residency expose this pair; the offload
    backends require it before allowing a component to stay in host memory.
    Mirrors the check the distributed layer-wise backend already makes.
    """
    return callable(getattr(module, "load_to_device", None)) and callable(getattr(module, "offload_to_cpu", None))


def model_level_vae_host_staging_requested(od_config: "OmniDiffusionConfig") -> bool:
    """Return whether the operator asked for host-staged VAEs at model level.

    Single source of truth for the policy half of the decision, so a pipeline
    deciding where to *construct* its VAEs and the backend deciding where to
    *keep* them cannot drift apart. The capability half -- the pipeline
    declaring the component on-demand and the component exposing the staging
    pair -- is checked separately by :func:`host_staged_component_names`.
    """
    return bool(getattr(od_config, "enable_cpu_offload", False) and getattr(od_config, "vae_cpu_offload", False))


def host_staged_component_names(
    pipeline: nn.Module,
    modules: list[nn.Module],
    names: list[str],
    *,
    policy_requested: bool,
) -> list[str]:
    """Return the components that may stay in host memory.

    Three conditions, all required: the operator asked for it, the pipeline
    declared the component as one it stages itself, and the component exposes
    the staging pair. Anything else keeps the resident placement, so a pipeline
    that cannot stage on demand degrades to the old behavior instead of failing
    later inside encode/decode.
    """
    if not policy_requested:
        return []
    from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan

    plan = get_offload_plan(pipeline)
    if plan is None:
        return []
    declared = plan.on_demand_component_paths
    return [name for module, name in zip(modules, names) if name in declared and can_stage_on_demand(module)]


class OffloadBackend(ABC):
    """Base class for CPU offload backends"""

    def __init__(self, config: OffloadConfig, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: nn.Module) -> None:
        """Enable offloading on the pipeline.

        Discovers modules, moves them to appropriate devices, and
        registers forward hooks for swapping/prefetching.

        Args:
            pipeline: Diffusion pipeline model (e.g., Wan22Pipeline)
        """
        raise NotImplementedError

    @abstractmethod
    def disable(self) -> None:
        """Disable offloading and cleanup resources.

        Removes all registered hooks. Does NOT move modules back to
        original devices (caller responsible for that).
        """
        raise NotImplementedError

    def is_enabled(self) -> bool:
        return self.enabled
