# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.data import OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_h3_profiler_targets_cover_aggregate_reference_encoders():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    targets = MiniMaxH3Pipeline._PROFILER_TARGETS
    assert "_encode_visual_conditions" in targets
    assert "_encode_reference_audio_conditions" in targets
    assert "_encode_video_conditions" not in targets
    assert "_encode_video_audio_conditions" not in targets
    assert "_encode_audio_conditions" not in targets


@pytest.mark.parametrize("distributed", [False, True])
def test_manual_component_offload_wins_over_requested_model_offload(distributed):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(
        enable_cpu_offload=True,
        enable_layerwise_offload=not distributed,
        enable_distributed_layerwise_offload=distributed,
    )
    pipeline._model_cpu_offload_modules = []
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected
    component = Mock()

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})
    with pipeline._component_on_device(component, family="vae"):
        component.load_to_device.assert_called_once_with()
        component.offload_to_cpu.assert_not_called()

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_called_once_with()
    component.offload_to_cpu.assert_called_once_with()


def test_requested_model_offload_without_backend_uses_resident_components():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=True,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
    )
    pipeline._model_cpu_offload_modules = []
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.encode_ids.assert_called_once()


def test_h3_model_cpu_offload_registers_direct_vae_stages(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = torch.nn.Linear(2, 2)
    pipeline.transformers_ref = torch.nn.Linear(2, 2)
    pipeline.text_encoder = torch.nn.Linear(2, 2)
    pipeline.video_vae = torch.nn.Linear(2, 2)
    pipeline.audio_vae = torch.nn.Linear(2, 2)
    apply_offload = Mock()
    remove_offload = Mock()
    monkeypatch.setattr(module, "apply_sequential_offload", apply_offload)
    monkeypatch.setattr(module, "remove_sequential_offload", remove_offload)

    pipeline.enable_omni_model_cpu_offload(
        device=torch.device("cpu"),
        pin_memory=False,
        use_hsdp=False,
    )

    dits = [pipeline.transformer, pipeline.transformers_ref]
    stages = [pipeline.text_encoder, pipeline.video_vae, pipeline.audio_vae]
    apply_offload.assert_called_once_with(
        dit_modules=dits,
        encoder_modules=stages,
        device=torch.device("cpu"),
        pin_memory=False,
        use_hsdp=False,
        offload_initial_dits=True,
    )

    pipeline.disable_omni_model_cpu_offload()

    remove_offload.assert_called_once_with([*dits, *stages])


@pytest.mark.parametrize("decode_fails", [False, True])
def test_h3_model_cpu_offload_scopes_direct_vae_call(monkeypatch, decode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    component = torch.nn.Linear(2, 2)
    events = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    monkeypatch.setattr(module, "sequential_offload_component", record_component)
    pipeline._model_cpu_offload_modules = [component]

    def decode():
        with pipeline._component_on_device(component, family="vae"):
            events.append(("decode", component))
            if decode_fails:
                raise RuntimeError("decode failed")

    if decode_fails:
        with pytest.raises(RuntimeError, match="decode failed"):
            decode()
    else:
        decode()

    assert events == [
        ("activate", component),
        ("decode", component),
        ("offload", component),
    ]


@pytest.mark.parametrize("encode_fails", [False, True])
def test_h3_model_cpu_offload_batches_visual_reference_scope(monkeypatch, encode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.video_vae = Mock()
    pipeline._model_cpu_offload_modules = [pipeline.video_vae]
    events = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    def encode_image(image):
        events.append(("encode", image.size))
        if encode_fails and image.width == 32:
            raise RuntimeError("encode failed")
        return torch.ones(image.width // 16, 4)

    pipeline.video_vae.encode_image.side_effect = encode_image
    monkeypatch.setattr(module, "sequential_offload_component", record_component)
    monkeypatch.setattr(module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(module, "_broadcast_tensor", lambda value, **kwargs: value)
    images = [Image.new("RGB", (16, 16)), Image.new("RGB", (32, 16))]

    if encode_fails:
        with pytest.raises(RuntimeError, match="encode failed"):
            pipeline._encode_visual_conditions(images, None, video_count=0)
    else:
        rows, shapes = pipeline._encode_visual_conditions(images, None, video_count=0)
        assert rows.shape == (3, 4)
        assert shapes == [(1, 1, 1), (1, 1, 2)]

    assert events == [
        ("activate", pipeline.video_vae),
        ("encode", (16, 16)),
        ("encode", (32, 16)),
        ("offload", pipeline.video_vae),
    ]


def test_h3_model_cpu_offload_shares_image_and_video_scope(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.video_vae = Mock()
    pipeline.video_vae.is_distributed_enabled.return_value = False
    pipeline._model_cpu_offload_modules = [pipeline.video_vae]
    events = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    def encode_image(image):
        events.append(("image", image.size))
        return torch.ones(1, 4)

    def encode_video(frames):
        events.append(("video", frames))
        return torch.ones(3, 4), (3, 2, 2)

    pipeline.video_vae.encode_image.side_effect = encode_image
    pipeline.video_vae.encode_video.side_effect = encode_video
    monkeypatch.setattr(module, "sequential_offload_component", record_component)
    monkeypatch.setattr(module, "_dit_rank_world", lambda: (None, 0, 1))
    monkeypatch.setattr(module, "_broadcast_tensor", lambda value, **kwargs: value)
    monkeypatch.setattr(module, "load_video_frames", lambda path: f"frames:{path}")
    images = [Image.new("RGB", (16, 16)), Image.new("RGB", (32, 16))]
    prepared_videos = [{"prepared_path": "reference.mp4"}]

    rows, shapes = pipeline._encode_visual_conditions(
        images,
        prepared_videos,
        video_count=1,
    )

    assert rows.shape == (5, 4)
    assert shapes == [(1, 1, 1), (1, 1, 2), (3, 2, 2)]
    assert events == [
        ("activate", pipeline.video_vae),
        ("image", (16, 16)),
        ("image", (32, 16)),
        ("video", "frames:reference.mp4"),
        ("offload", pipeline.video_vae),
    ]


@pytest.mark.parametrize("encode_fails", [False, True])
def test_h3_model_cpu_offload_shares_embedded_and_standalone_audio_scope(monkeypatch, encode_fails):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.audio_vae = Mock()
    pipeline._model_cpu_offload_modules = [pipeline.audio_vae]
    events = []

    @contextmanager
    def record_component(value):
        events.append(("activate", value))
        try:
            yield
        finally:
            events.append(("offload", value))

    def encode_embedded(*args, **kwargs):
        events.append(("embedded",))
        if encode_fails:
            raise RuntimeError("audio encode failed")
        return torch.ones(1, 2), [1]

    def encode_standalone(*args, **kwargs):
        events.append(("standalone",))
        return torch.ones(1, 2), [1]

    pipeline._encode_video_audio_conditions_resident = encode_embedded
    pipeline._encode_audio_conditions_resident = encode_standalone
    monkeypatch.setattr(module, "sequential_offload_component", record_component)

    def call():
        return pipeline._encode_reference_audio_conditions(
            [{"input_has_audio": True}],
            has_audio=[True],
            standalone_audios=[(torch.ones(1), 16_000)],
            max_duration_seconds=1.0,
        )

    if encode_fails:
        with pytest.raises(RuntimeError, match="audio encode failed"):
            call()
    else:
        result = call()
        assert result[0].shape == (1, 2)
        assert result[2].shape == (1, 2)

    assert events[0] == ("activate", pipeline.audio_vae)
    assert events[-1] == ("offload", pipeline.audio_vae)
    assert events.count(("activate", pipeline.audio_vae)) == 1
    assert events.count(("offload", pipeline.audio_vae)) == 1


# ---------------------------------------------------------------------------
# Layer-wise component selection: the pipeline side of the contract.
#
# ``--layerwise-offload-components`` names the families the layer-wise backend
# may manage; a family left out stays fully device-resident. This pipeline
# stages the encoders and VAEs itself, so it has to read the same selection --
# otherwise it drags an unselected family back to the host after every use and
# the placement the backend published never holds.
# ---------------------------------------------------------------------------


def _residency_pipeline(**od_kwargs):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(**od_kwargs)
    pipeline._model_cpu_offload_modules = []
    pipeline.device = torch.device("cpu")
    return pipeline


def test_unselected_vae_family_is_not_staged():
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,text_encoder",
    )
    component = Mock()

    with pipeline._component_on_device(component, family="vae"):
        pass

    assert pipeline._stages_component_family("vae") is False
    component.load_to_device.assert_not_called()
    component.offload_to_cpu.assert_not_called()


def test_unselected_vae_family_is_built_on_the_device():
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,text_encoder",
    )

    # Constructing on the host would only buy a start-up host-to-device
    # transfer for weights the backend then keeps resident.
    assert pipeline._vae_load_device() == pipeline.device


def test_default_selection_still_stages_every_family():
    # The lock on "default behavior is byte-for-byte unchanged": omitting the
    # option selects every family, so both sites behave exactly as before.
    pipeline = _residency_pipeline(enable_layerwise_offload=True)
    component = Mock()

    with pipeline._component_on_device(component, family="vae"):
        component.load_to_device.assert_called_once_with()
        component.offload_to_cpu.assert_not_called()

    component.offload_to_cpu.assert_called_once_with()
    assert pipeline._stages_component_family("text_encoder") is True
    assert pipeline._vae_load_device() == torch.device("cpu")


def test_distributed_layerwise_stages_families_the_selection_omits():
    # Distributed layer-wise offload shards every component it manages; its
    # residency does not go through the plain-layerwise selection.
    pipeline = _residency_pipeline(
        enable_distributed_layerwise_offload=True,
        layerwise_offload_components="dit",
    )
    component = Mock()

    with pipeline._component_on_device(component, family="vae"):
        component.load_to_device.assert_called_once_with()

    component.offload_to_cpu.assert_called_once_with()
    assert pipeline._vae_load_device() == torch.device("cpu")


def test_unselected_text_encoder_family_skips_the_host_round_trip():
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,vae",
    )
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    # Resident: loaded once, never pushed back to the host.
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_not_called()


def test_selected_text_encoder_family_keeps_the_host_round_trip():
    pipeline = _residency_pipeline(
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,text_encoder",
    )
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_called_once_with()
