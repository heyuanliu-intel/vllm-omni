# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 initial-noise materialization.

Pure-function extraction of the tensor block in the checkpoint's
``MiniMaxH3LatentPreparationStage._prepare_denoise_state_from_plan``. The stage
class itself is bound to the SGLang ``Req``/plan objects, so only the noise
recipe is carried over here -- statement for statement, in the same order.

Noise semantics (from that stage's own comment, reproduced because they are a
contract and not a convention):
- video noise is drawn on the RAW latent tensor [1, 24, T, H_lat, W_lat] in
  tensor layout, then patchified into packed row order;
- audio uses an INDEPENDENT generator re-seeded with the same seed (each
  modality re-seeds its own generator);
- no extra cond-frame noise is drawn for image-conditioned requests.

Drawing audio from the video generator's continuing stream, or drawing it in
[C, D, T] layout and permuting, both produce a different sample for the same
seed -- valid-looking noise that does not reproduce the reference output.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)

MINIMAX_H3_VIDEO_LATENT_CHANNELS = 24
MINIMAX_H3_AUDIO_LATENT_CHANNELS = 32
MINIMAX_H3_AUDIO_CHANNELS = 2
MINIMAX_H3_PATCH_SIZE = [1, 2, 2]


def minimax_h3_initial_noise(
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Draw the packed initial video/audio noise rows for one request."""
    latent_t = int(latent_t)
    latent_h = int(latent_h)
    latent_w = int(latent_w)
    audio_t = int(audio_t)

    video_rows_n = latent_t * (latent_h // 2) * (latent_w // 2)
    audio_rows_n = audio_t * MINIMAX_H3_AUDIO_CHANNELS

    gen_v = torch.Generator().manual_seed(int(seed))
    video_tensor = torch.randn(
        1,
        MINIMAX_H3_VIDEO_LATENT_CHANNELS,
        latent_t,
        latent_h,
        latent_w,
        generator=gen_v,
        dtype=torch.float32,
    )
    video_noise = minimax_h3_patchify_video_latent(
        video_tensor, patch_size=MINIMAX_H3_PATCH_SIZE
    ).to(torch.float32)
    gen_a = torch.Generator().manual_seed(int(seed))
    audio_noise = torch.randn(
        audio_rows_n,
        MINIMAX_H3_AUDIO_LATENT_CHANNELS,
        generator=gen_a,
        dtype=torch.float32,
    )
    if list(video_noise.shape) != [video_rows_n, 96]:
        raise ValueError(
            f"aligned video noise shape {list(video_noise.shape)} != "
            f"[{video_rows_n}, 96]"
        )
    return {
        "initial_video_rows": video_noise,
        "initial_audio_rows": audio_noise,
        "latent_t": latent_t,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "audio_t": audio_t,
    }


__all__ = ["minimax_h3_initial_noise"]
