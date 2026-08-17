# SPDX-License-Identifier: Apache-2.0
"""FL2VA dual-keyframe (first/last) request wiring.

Covers the request-side contract that unlocks the upstream
MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES ((0,), (-1,), (0, -1)) from the
pipeline hardcode, without touching GPU paths.
"""

import pytest
import torch
from PIL import Image


def test_resolve_fl2va_keyframe_indices_defaults_and_overrides():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_fl2va_keyframe_indices,
    )

    assert _resolve_fl2va_keyframe_indices({}, 1) == [0]
    assert _resolve_fl2va_keyframe_indices({}, 2) == [0, -1]
    assert _resolve_fl2va_keyframe_indices({"frame_index": -1}, 1) == [-1]
    assert _resolve_fl2va_keyframe_indices({"frame_indices": [0, -1]}, 2) == [0, -1]
    assert _resolve_fl2va_keyframe_indices({"target": {"frame_index": -1}}, 1) == [-1]


def test_resolve_fl2va_keyframe_indices_rejects_bad_input():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_fl2va_keyframe_indices,
    )

    with pytest.raises(ValueError, match="frame_indices"):
        _resolve_fl2va_keyframe_indices({"frame_indices": [0, 1]}, 2)
    with pytest.raises(ValueError, match="frame_indices"):
        _resolve_fl2va_keyframe_indices({"frame_indices": [-1, 0]}, 2)
    with pytest.raises(ValueError, match="one frame index per image"):
        _resolve_fl2va_keyframe_indices({"frame_indices": [0]}, 2)
    with pytest.raises(ValueError, match="one frame index per image"):
        _resolve_fl2va_keyframe_indices({"frame_indices": [0, -1]}, 1)
    with pytest.raises(ValueError, match="must contain only integers"):
        _resolve_fl2va_keyframe_indices({"frame_indices": [0, True]}, 2)


def test_load_images_accepts_single_and_pair():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _load_images

    first = Image.new("RGB", (8, 8))
    last = Image.new("RGB", (8, 8))
    assert len(_load_images(first)) == 1
    assert len(_load_images([first, last])) == 2
    with pytest.raises(ValueError, match="must not be empty"):
        _load_images([])


@pytest.mark.parametrize(
    ("frame_indices", "expected_cond_rows"),
    [
        ([0], 6),
        ([-1], 6),
        ([0, -1], 12),
    ],
)
def test_fl2va_packing_all_keyframe_signatures(frame_indices, expected_cond_rows):
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    # latent_h=4 x latent_w=6 with the fixed [1,2,2] patch => 6 rows per frame.
    packed = minimax_h3_packed_sequence(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        include_keyframe_cond=True,
        keyframe_frame_indices=frame_indices,
        frame_count=5,
    )
    video_rows = 2 * 6
    assert packed["img_pos"].numel() == video_rows + expected_cond_rows
    assert packed["update_mask"].sum().item() == video_rows
    assert (~packed["update_mask"]).sum().item() == expected_cond_rows


def test_fl2va_packing_rejects_unsupported_signatures():
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    common = dict(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        include_keyframe_cond=True,
        frame_count=5,
    )
    with pytest.raises(ValueError, match="keyframe_frame_indices in"):
        minimax_h3_packed_sequence(**common, keyframe_frame_indices=[1])
    with pytest.raises(ValueError, match="keyframe_frame_indices in"):
        minimax_h3_packed_sequence(**common, keyframe_frame_indices=[-1, 0])


def test_fl2va_single_keyframe_packing_unchanged_by_dual_support():
    """Regression pin: the (0,) layout must be identical to the historical
    hardcoded-[0] behavior (same numbers as the pre-change packing test)."""
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    fl2va = minimax_h3_packed_sequence(
        text_len=4,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0],
        frame_count=5,
    )
    assert fl2va["img_pos"].numel() == 18
    assert fl2va["update_mask"].sum().item() == 12
    assert (~fl2va["update_mask"]).sum().item() == 6
    assert fl2va["audio_pos"].numel() == 6


def test_dual_keyframe_cond_blocks_are_ordered_first_then_last():
    """The two cond blocks must bind frame 0 then the last frame, matching
    the image order sent by the request side."""
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
        _resolve_keyframe_frame_indices,
    )

    assert _resolve_keyframe_frame_indices([0, -1], frame_count=5) == [0, 4]
    assert _resolve_keyframe_frame_indices([-1], frame_count=5) == [4]
    with pytest.raises(ValueError, match="already bound"):
        _resolve_keyframe_frame_indices([0, 0], frame_count=5)


def test_multi_image_presentation_labels_two_pictures():
    """Two keyframes produce two labeled vision slots in the fl2va
    presentation, in image order (Picture 1 = first, Picture 2 = last)."""
    from vllm_omni.diffusion.models.minimax_h3.presentation import (
        _multi_image_presentation,
    )

    class _FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(ch) % 97 for ch in text]}

        def convert_tokens_to_ids(self, token):
            return 7

    single_ids, single_tags = _multi_image_presentation(
        _FakeTokenizer(), prompt="p", image_token_counts=[3]
    )
    dual_ids, dual_tags = _multi_image_presentation(
        _FakeTokenizer(), prompt="p", image_token_counts=[3, 5]
    )
    assert dual_ids.shape[0] > single_ids.shape[0]
    assert dual_tags.shape[0] == dual_ids.shape[0]
    # the extra slot carries the second image's tokens plus its label text
    assert dual_ids.shape[0] - single_ids.shape[0] >= 5
