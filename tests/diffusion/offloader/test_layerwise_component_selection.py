# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Layer-wise offloading must honor the component-family selection.

``layerwise_offload_components`` names which families the backend may manage;
a family left out stays fully device-resident. Excluding ``dit`` is the
load-bearing case: it yields the topology where the DiT fits whole on the
device and only the other components are offloaded (the configuration the
sglang-side runs use for TP4xSP2), instead of paying the per-block streaming
cost on the component that dominates step time.
"""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_selection_defaults_to_every_family() -> None:
    od = OmniDiffusionConfig(enable_layerwise_offload=True)

    assert od.layerwise_component_selection() == frozenset({"dit", "text_encoder", "vae"}), (
        "None must select every family -- that is the pre-existing behavior"
    )


def test_selection_parses_comma_list() -> None:
    od = OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components=" text_encoder, vae ")

    assert od.layerwise_component_selection() == frozenset({"text_encoder", "vae"})


@pytest.mark.parametrize("bad", ["ditt", "dit,encoder", "", " , "])
def test_selection_rejects_unknown_or_empty(bad: str) -> None:
    # A typo must fail loudly: silently narrowing the selection would change
    # which components stay resident, and that is a capacity decision.
    with pytest.raises(ValueError, match="layerwise_offload_components"):
        OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components=bad)


def test_offload_config_carries_selection() -> None:
    od = OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components="text_encoder,vae")

    cfg = OffloadConfig.from_od_config(od)

    assert cfg.layerwise_components == frozenset({"text_encoder", "vae"})


def test_offload_config_defaults_to_every_family() -> None:
    cfg = OffloadConfig.from_od_config(OmniDiffusionConfig(enable_layerwise_offload=True))

    assert cfg.layerwise_components == frozenset({"dit", "text_encoder", "vae"})


class _StageableVAE(nn.Module):
    """A VAE the pipeline declares on-demand: it stages itself to the host."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.offload_calls = 0
        self.load_calls = 0

    def offload_to_cpu(self) -> None:
        self.offload_calls += 1

    def load_to_device(self) -> None:
        self.load_calls += 1


class _Pipeline(nn.Module):
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["video_vae"]
    _resident_modules = []

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)
        self.video_vae = _StageableVAE()
        self._offload_plan = OffloadPlan(
            encoder_block_attrs={},
            on_demand_component_paths=frozenset({"video_vae"}),
        )


def _backend(components: str | None) -> LayerWiseOffloadBackend:
    # Bypass __init__: it opens a device copy stream, which does not exist in a
    # CPU-only environment, and the gating paths under test never touch it.
    backend = object.__new__(LayerWiseOffloadBackend)
    backend.config = OffloadConfig.from_od_config(
        OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components=components)
    )
    backend.device = torch.device("cpu")
    backend.enabled = False
    backend._blocks = []
    backend._streamed_encoders = []
    return backend


@pytest.fixture()
def encoder_streams(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record which encoders production hands to block streaming."""
    import vllm_omni.diffusion.offloader.layerwise_backend as backend_module

    streamed: list[str] = []

    def _record(enc, enc_name, plan, device, *, pin_memory=True):
        streamed.append(enc_name)
        return False

    monkeypatch.setattr(backend_module, "stream_declared_encoder_blocks", _record)
    return streamed


def test_enable_keeps_excluded_dit_resident_without_hooks(encoder_streams: list[str]) -> None:
    backend = _backend("text_encoder,vae")
    pipeline = _Pipeline()

    backend.enable(pipeline)

    assert backend.enabled is True, "the backend still manages the remaining families"
    assert getattr(pipeline.transformer, "_hook_registry", None) is None, (
        "an excluded DiT must not receive block-streaming hooks"
    )
    assert backend._blocks == [], "no block group may be tracked for an excluded DiT"
    assert encoder_streams == ["text_encoder"], "the selected encoder family is still streamed"
    assert pipeline.video_vae.offload_calls == 1, "the selected VAE family is still host-staged"


def test_enable_keeps_excluded_encoder_resident(encoder_streams: list[str]) -> None:
    backend = _backend("dit,vae")
    pipeline = _Pipeline()

    backend.enable(pipeline)

    assert encoder_streams == [], "an excluded encoder family must not be handed to block streaming"


def test_enable_keeps_excluded_vae_resident(encoder_streams: list[str]) -> None:
    backend = _backend("dit,text_encoder")
    pipeline = _Pipeline()

    backend.enable(pipeline)

    assert pipeline.video_vae.offload_calls == 0, (
        "an excluded VAE family must stay device-resident, not host-staged"
    )


def test_serve_cli_registers_the_flag_and_it_reaches_the_stage_config() -> None:
    # The serve parser is a hand-written argument table, not generated from the
    # dataclass fields: a field alone leaves the documented flag unregistered
    # and the CLI rejects it before a server ever starts.
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        ["serve", "fake-model", "--omni", "--layerwise-offload-components", "text_encoder,vae"]
    )
    explicit = args.get_explicit_kwargs_dict()
    assert explicit["layerwise_offload_components"] == "text_encoder,vae"

    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {"model": "fake-model", "layerwise_offload_components": explicit["layerwise_offload_components"]}
    )
    assert stage_cfg[0]["engine_args"]["layerwise_offload_components"] == "text_encoder,vae"


def test_typed_deploy_config_accepts_the_field() -> None:
    from vllm_omni.config.stage_config import StageDeployConfig

    cfg = StageDeployConfig(stage_id=0, layerwise_offload_components="text_encoder,vae")

    assert cfg.layerwise_offload_components == "text_encoder,vae"


class _CountingEncoder:
    """Stands in for the H3 text encoder: the request path must not evict an
    excluded family after use."""

    def __init__(self) -> None:
        self.loads = 0
        self.offloads = 0

    def load_to_device(self) -> None:
        self.loads += 1

    def offload_to_cpu(self) -> None:
        self.offloads += 1

    def encode_ids(self, input_ids, **vision_kwargs):
        return input_ids


class _H3Stub:
    def __init__(self, components: str | None):
        self.od_config = OmniDiffusionConfig(
            enable_layerwise_offload=True, layerwise_offload_components=components
        )
        self.text_encoder = _CountingEncoder()


def _h3_pipeline_cls():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    return MiniMaxH3Pipeline


def test_encode_text_hidden_keeps_excluded_encoder_resident() -> None:
    # The contract is the whole request lifecycle, not just startup placement:
    # an excluded encoder must not be evicted back to the host after use.
    cls = _h3_pipeline_cls()
    stub = _H3Stub("dit,vae")
    ids = torch.zeros(1, dtype=torch.long)

    out = cls._encode_text_hidden(stub, ids, {})

    assert out is ids
    assert stub.text_encoder.offloads == 0, "an excluded encoder must never be offloaded on the request path"
    assert stub.text_encoder.loads == 1


def test_encode_text_hidden_streams_selected_encoder() -> None:
    cls = _h3_pipeline_cls()
    stub = _H3Stub(None)
    ids = torch.zeros(1, dtype=torch.long)

    cls._encode_text_hidden(stub, ids, {})

    assert stub.text_encoder.loads == 1 and stub.text_encoder.offloads == 1


def test_component_on_device_keeps_excluded_vae_resident() -> None:
    cls = _h3_pipeline_cls()
    stub = _H3Stub("dit,text_encoder")
    stub._vae_staging_active = cls._vae_staging_active.__get__(stub)
    vae = _CountingEncoder()

    with cls._component_on_device(stub, vae):
        pass

    assert vae.loads == 0 and vae.offloads == 0, (
        "an excluded VAE must not be staged in and out around its use"
    )


def test_component_on_device_stages_selected_vae() -> None:
    cls = _h3_pipeline_cls()
    stub = _H3Stub(None)
    stub._vae_staging_active = cls._vae_staging_active.__get__(stub)
    vae = _CountingEncoder()

    with cls._component_on_device(stub, vae):
        pass

    assert vae.loads == 1 and vae.offloads == 1


def test_typed_deploy_chain_carries_the_field_into_the_projection() -> None:
    # The typed deploy path is StageDeployConfig -> _stage_engine_overrides ->
    # _DiffusionConfigProjection; from_kwargs silently filters unknown fields,
    # so a missing projection field is a real break, not a no-op.
    from vllm_omni.config.omni_config import _DiffusionConfigProjection, _stage_engine_overrides
    from vllm_omni.config.stage_config import StageDeployConfig

    overrides = _stage_engine_overrides(StageDeployConfig(stage_id=0, layerwise_offload_components="text_encoder,vae"))
    assert overrides["layerwise_offload_components"] == "text_encoder,vae"

    projection = _DiffusionConfigProjection.from_kwargs(**overrides)
    assert projection.layerwise_offload_components == "text_encoder,vae"

    empty = _DiffusionConfigProjection.from_kwargs(**_stage_engine_overrides(StageDeployConfig(stage_id=0)))
    assert empty.layerwise_offload_components is None, "an unset selection must not be invented"


class _LogRecorder:
    def __init__(self) -> None:
        self.warnings: list[tuple] = []

    def warning(self, msg, *args) -> None:
        self.warnings.append((msg, args))

    def info(self, msg, *args) -> None:
        pass


def test_other_strategies_warn_when_selection_is_narrowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.diffusion.offloader.base as base_module

    log = _LogRecorder()
    monkeypatch.setattr(base_module, "logger", log)

    OffloadConfig.from_od_config(
        OmniDiffusionConfig(enable_distributed_layerwise_offload=True, layerwise_offload_components="dit")
    )

    assert any("plain layer-wise" in str(msg) for msg, _ in log.warnings), (
        "a narrowed selection under a non-consuming strategy must be called out"
    )


def test_vae_staging_resolver_follows_strategy_priority() -> None:
    # One resolver serves both the VAE construction site and the per-use scope;
    # it must resolve in the offload backend's priority: distributed layer-wise
    # (which ignores the selection), then plain layer-wise (which consumes it),
    # then the model-level host-staging policy.
    cls = _h3_pipeline_cls()

    def active(**kwargs) -> bool:
        stub = type("_S", (), {})()
        stub.od_config = OmniDiffusionConfig(**kwargs)
        return cls._vae_staging_active(stub)

    assert active(enable_distributed_layerwise_offload=True, layerwise_offload_components="dit") is True, (
        "DLO always stages, whatever the selection says"
    )
    assert active(enable_layerwise_offload=True, layerwise_offload_components="dit,text_encoder") is False
    assert (
        active(
            enable_layerwise_offload=True,
            enable_cpu_offload=True,
            vae_cpu_offload=True,
            layerwise_offload_components="dit,text_encoder",
        )
        is False
    ), "plain layer-wise wins the priority: its selection overrides the model-level host-staging policy"
    assert active(enable_cpu_offload=True, vae_cpu_offload=True) is True, (
        "without layer-wise the model-level host-staging policy still decides"
    )


class _CallableEncoder(_CountingEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __call__(self, input_ids, **vision_kwargs):
        self.calls += 1
        return input_ids


def _encode_with(**config_kwargs):
    cls = _h3_pipeline_cls()
    stub = type("_S", (), {})()
    stub.od_config = OmniDiffusionConfig(**config_kwargs)
    stub.text_encoder = _CallableEncoder()
    cls._encode_text_hidden(stub, torch.zeros(1, dtype=torch.long), {})
    return stub.text_encoder


def test_encode_text_hidden_streams_under_eco_plus_layerwise() -> None:
    # Plain layer-wise is the active strategy when both flags are set; the
    # encode path must stream, not fall into the model-level branch.
    enc = _encode_with(enable_cpu_offload=True, enable_layerwise_offload=True)

    assert enc.loads == 1 and enc.offloads == 1 and enc.calls == 0


def test_encode_text_hidden_streams_under_dlo() -> None:
    enc = _encode_with(enable_distributed_layerwise_offload=True)

    assert enc.loads == 1 and enc.offloads == 1 and enc.calls == 0, (
        "DLO must keep the low-residency encoder phase"
    )


def test_encode_text_hidden_uses_model_level_branch_without_layerwise() -> None:
    enc = _encode_with(enable_cpu_offload=True)

    assert enc.calls == 1 and enc.loads == 0 and enc.offloads == 0, (
        "model-level offload routes through nn.Module.__call__ so the swap hooks run"
    )


def test_encode_text_hidden_eco_plus_layerwise_excluded_te_stays_resident() -> None:
    # With plain layer-wise active, model-level offload is disabled by the
    # strategy priority -- so an encoder excluded from the selection must take
    # the resident branch, not fall through to the model-level swap.
    enc = _encode_with(
        enable_cpu_offload=True,
        enable_layerwise_offload=True,
        layerwise_offload_components="dit,vae",
    )

    assert enc.loads == 1 and enc.offloads == 0 and enc.calls == 0
