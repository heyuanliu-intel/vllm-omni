# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-H3 DiT architecture config.

Field values are transcribed from the checkpoint's own
`transformer/config.json` and from SGLang's
`configs/models/dits/minimax_h3.py` (commit 5d8a20b17). Do not "tidy" these
numbers — they must match the weights exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.diffusion.models.minimax_h3.compat import AttentionBackendEnum


@dataclass
class MiniMaxH3DiTArchConfig:
    # --- fields inherited from SGLang's DiTArchConfig -----------------------
    _fsdp_shard_conditions: list = field(default_factory=list)
    _compile_conditions: list = field(default_factory=list)
    param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    exclude_lora_layers: list[str] = field(default_factory=list)
    boundary_ratio: float | None = None

    # H3 ships CFG-distilled checkpoints and only ever ran on FlashAttention.
    # On XPU this resolves to vLLM-Omni's FlashAttentionBackend, whose varlen
    # wrapper covers both XPU and CUDA.
    _supported_attention_backends: set = field(
        default_factory=lambda: {AttentionBackendEnum.FA}
    )

    # --- H3 architecture ----------------------------------------------------
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24          # video latent channels
    audio_latents_dim: int = 32    # audio latent channels (joint stream)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376        # 96768
    final_adaln_out_features: int = 2 * 5376   # 10752
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    # set in __post_init__
    num_channels_latents: int = 0

    def __post_init__(self) -> None:
        if not self._compile_conditions:
            self._compile_conditions = list(self._fsdp_shard_conditions)
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values, got {self.patch_size}.")
        self.num_channels_latents = self.latents_dim

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> MiniMaxH3DiTArchConfig:
        """Build the arch from the checkpoint's own transformer/config.json.

        Any field the mapping does not carry keeps the default transcribed
        above. Verified against the FL2VA checkpoint: all 19 keys it declares
        match the defaults exactly, so this changes nothing today -- it stops
        a future checkpoint with a different architecture from being loaded
        silently under the wrong shape.
        """
        fields = cls.__dataclass_fields__
        values = {name: config[name] for name in fields if name in config}
        if "patch_size" in values:
            values["patch_size"] = tuple(values["patch_size"])
        arch = cls(**values)
        if len(arch.patch_size) != 3:
            raise ValueError(
                f"patch_size must contain three values, got {arch.patch_size!r}"
            )
        return arch


@dataclass
class MiniMaxH3DiTConfig:
    arch_config: MiniMaxH3DiTArchConfig = field(
        default_factory=MiniMaxH3DiTArchConfig
    )
    prefix: str = ""
    quant_config: object | None = None
    torch_compile_mode: str = "max-autotune-no-cudagraphs"


__all__ = ["MiniMaxH3DiTArchConfig", "MiniMaxH3DiTConfig"]
