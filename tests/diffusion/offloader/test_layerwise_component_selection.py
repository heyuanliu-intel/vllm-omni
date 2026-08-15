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


class _Pipeline(nn.Module):
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = []
    _resident_modules = []

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)


def _backend(components: str | None) -> LayerWiseOffloadBackend:
    # Bypass __init__: it opens a device copy stream, which does not exist in a
    # CPU-only environment, and the excluded-DiT path never touches it.
    backend = object.__new__(LayerWiseOffloadBackend)
    backend.config = OffloadConfig.from_od_config(
        OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components=components)
    )
    backend.device = torch.device("cpu")
    backend.enabled = False
    backend._blocks = []
    return backend


def test_enable_keeps_excluded_dit_resident_without_hooks() -> None:
    backend = _backend("text_encoder,vae")
    pipeline = _Pipeline()

    backend.enable(pipeline)

    assert backend.enabled is True, "the backend still manages the remaining families"
    assert getattr(pipeline.transformer, "_hook_registry", None) is None, (
        "an excluded DiT must not receive block-streaming hooks"
    )
    assert backend._blocks == [], "no block group may be tracked for an excluded DiT"


def test_serve_cli_registers_flag_and_carries_it_to_default_stage() -> None:
    # The serve parser is hand-written. A dataclass field alone would leave the
    # public flag unregistered and fail before the server starts.
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


def test_typed_deploy_chain_carries_selection_to_projection() -> None:
    # Projection silently filters unknown fields, so exercise the complete
    # typed deployment chain rather than checking only the source dataclass.
    from vllm_omni.config.omni_config import _DiffusionConfigProjection, _stage_engine_overrides
    from vllm_omni.config.stage_config import StageDeployConfig

    overrides = _stage_engine_overrides(
        StageDeployConfig(stage_id=0, layerwise_offload_components="text_encoder,vae")
    )
    assert overrides["layerwise_offload_components"] == "text_encoder,vae"

    projection = _DiffusionConfigProjection.from_kwargs(**overrides)
    assert projection.layerwise_offload_components == "text_encoder,vae"

    empty = _DiffusionConfigProjection.from_kwargs(**_stage_engine_overrides(StageDeployConfig(stage_id=0)))
    assert empty.layerwise_offload_components is None
