# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ulysses all-to-all placeholders.

This port targets TP with SP=1, and upstream rejects TP+SP together, so the
DiT never enters its `sp_active` branch. These exist so the (function-local)
import site stays resolvable; calling them is a bug, not a fallback.
"""


def _usp_input_all_to_all_varlen(*args, **kwargs):
    raise RuntimeError(
        "Ulysses sequence parallelism is not wired in the vLLM-Omni MiniMax-H3 "
        "port (TP with SP=1 only)."
    )


def _usp_output_all_to_all_varlen(*args, **kwargs):
    raise RuntimeError(
        "Ulysses sequence parallelism is not wired in the vLLM-Omni MiniMax-H3 "
        "port (TP with SP=1 only)."
    )
