# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capabilities specific to the MiniMax-H3 video pipeline."""


def minimax_h3_preserves_reference_image_size(*, model: str | None = None, revision: str | None = None) -> bool:
    """Report that MiniMax-H3 owns reference-image geometry.

    The pipeline rescales reference images itself and then validates the
    geometry it was handed. Resizing the image to the requested output size in
    the serving layer would distort what the pipeline encodes and make those
    checks describe the resized copy instead of what the client sent, so the
    serving layer must forward the client's original image untouched.
    """
    return True
