# SPDX-License-Identifier: Apache-2.0
"""Upstream MiniMax-H3 modules reused verbatim.

Copied unchanged from SGLang-Diffusion (commit 5d8a20b17). They
import only torch/math/typing, so they carry no framework coupling
and need no edits. Do not hand-edit anything in this package: the
point is that it stays byte-identical to upstream, so contracts
like the 5n+2 video latent T and the 40 Hz audio latent rate
cannot drift through transcription.

Two files are held to a weaker but explicit standard, because their
upstream form is a pipeline-stage class bound to SGLang's `Req` and
cannot be imported here:

- `denoise_loop.py` -- body byte-identical; only the scheduler import
  path is rewritten.
- `initial_noise.py` -- the noise block of
  `MiniMaxH3LatentPreparationStage._prepare_denoise_state_from_plan`,
  carried over statement for statement as a pure function. It is
  pinned by the checkpoint's own `test_t2va_seed_noise_recipe`
  numbers in `test_h3_contract.py`.

Anything else in here must stay a straight copy.
"""
