# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility shim so the MiniMax-H3 DiT can be ported near-verbatim.

The upstream DiT (SGLang `multimodal_gen`, commit 5d8a20b17) depends on only
seven SGLang-internal symbols. Rather than rewrite 1100 lines of numerically
sensitive model code — where a subtle transcription error would show up as
slightly degraded video rather than a crash — we re-expose those seven names
on top of vLLM / vLLM-Omni equivalents and keep the model body intact.

Mapping:
  get_tp_world_size        -> vllm.distributed (falls back to 1 when the
                              process group is not initialised)
  ColumnParallelLinear     -> vllm.model_executor.layers.linear (same names)
  MergedColumnParallelLinear    ""      (this is what makes TP>1 possible:
                              it shards packed qkv / fc1 per logical matrix)
  RowParallelLinear             ""
  QuantizationConfig       -> vllm ... quantization.base_config
  AttentionBackendEnum     -> local enum; only `.FA` is ever compared against
  get_attn_backend         -> resolves through vLLM-Omni's platform
  CachableDiT              -> plain nn.Module base (TeaCache is an optional
                              upstream optimisation the H3 DiT body never calls)

SP note: H3's Ulysses path is mutually exclusive with TP (upstream raises when
both are on). This port targets TP, so the Ulysses hooks below report a world
size of 1, which makes the DiT take its non-SP branch.
"""

from __future__ import annotations

import enum
from typing import Any

import torch
from torch import nn

# --- linear layers: same class names, so the model body needs no edits -------
from vllm.model_executor.layers.linear import (  # noqa: F401
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: F401
    QuantizationConfig,
)


def get_tp_world_size() -> int:
    """TP world size, or 1 when distributed is not initialised."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def get_tp_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


class AttentionBackendEnum(enum.Enum):
    """Subset of SGLang's enum. The H3 DiT only ever compares against FA."""

    FA = enum.auto()
    FA2 = enum.auto()
    TORCH_SDPA = enum.auto()

    def get_enum(self) -> AttentionBackendEnum:
        return self


class _ResolvedBackend:
    """Mimics SGLang's selector return value, which exposes .get_enum()."""

    def __init__(self, backend: AttentionBackendEnum):
        self._backend = backend

    def get_enum(self) -> AttentionBackendEnum:
        return self._backend


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: set | None = None,
) -> _ResolvedBackend:
    """Resolve the diffusion attention backend through the vLLM-Omni platform.

    On Intel XPU this yields FlashAttentionBackend, whose varlen wrapper
    (`diffusion.attention.backends.utils.fa.flash_attn_varlen_func`) covers
    XPU and CUDA behind one signature — which is exactly what the H3
    attention body calls.
    """
    try:
        from vllm_omni.platforms import current_omni_platform

        path = current_omni_platform.get_diffusion_attn_backend_cls(None, head_size)
        if "flash_attn" in path.lower() or "flashattention" in path.lower():
            return _ResolvedBackend(AttentionBackendEnum.FA)
        return _ResolvedBackend(AttentionBackendEnum.TORCH_SDPA)
    except Exception:
        # Keep the model loadable off-device (e.g. meta-device shape checks).
        return _ResolvedBackend(AttentionBackendEnum.FA)


# --- Ulysses / SP hooks: this port runs TP with SP=1 -------------------------

def model_parallel_is_initialized() -> bool:
    """Reported as False so the DiT takes its SP=1 branch.

    Upstream `_ulysses_ctx()` short-circuits to (world_size=1, rank=0) when this
    is False, which is exactly the non-Ulysses path this TP-targeted port wants.
    TP is read separately via get_tp_world_size(), so this does not disable TP.
    """
    return False


def get_ulysses_parallel_world_size() -> int:
    return 1


def get_ulysses_parallel_rank() -> int:
    return 0


def get_sp_group():
    raise RuntimeError(
        "MiniMax-H3 in vLLM-Omni is ported for TP with SP=1; the Ulysses "
        "sequence-parallel path is not wired. Upstream also rejects TP+SP "
        "together."
    )


# --- DiT base ---------------------------------------------------------------

class CachableDiT(nn.Module):
    """Stand-in for SGLang's CachableDiT.

    Upstream layers TeaCache onto BaseDiT. The H3 DiT body never calls the
    cache hooks, so an nn.Module that records config/hf_config is behaviourally
    equivalent for inference while dropping a large dependency tail.
    """

    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    param_names_mapping: dict = {}
    reverse_param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}
    _supported_attention_backends: set = {AttentionBackendEnum.FA}

    hidden_size: int
    num_attention_heads: int
    num_channels_latents: int

    def __init__(self, config: Any = None, hf_config: Any = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.hf_config = hf_config


__all__ = [
    "AttentionBackendEnum",
    "CachableDiT",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QuantizationConfig",
    "RowParallelLinear",
    "get_attn_backend",
    "get_sp_group",
    "get_tp_rank",
    "get_tp_world_size",
    "get_ulysses_parallel_rank",
    "get_ulysses_parallel_world_size",
    "model_parallel_is_initialized",
]
