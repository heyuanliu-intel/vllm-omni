# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for SequentialOffloadBackend."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend, SequentialOffloadHook
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture
def accelerator_device() -> torch.device:
    """Fixture that provides accelerator device or skips test if unavailable."""
    if current_omni_platform.get_device_count() == 0:
        pytest.skip("Accelerator required for this test")
    return current_omni_platform.get_torch_device(0)


def _create_simple_module() -> nn.Module:
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)

    return SimpleModule()


def _track_pin_memory_calls():
    tracker = {"called": False}
    original = torch.Tensor.pin_memory

    def mock(self):
        tracker["called"] = True
        return original(self)

    return tracker, mock


def test_model_level_backend_delegates_to_custom_pipeline_offload() -> None:
    class CustomPipeline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enable_args = None
            self.disable_called = False

        def enable_omni_model_cpu_offload(self, **kwargs) -> None:
            self.enable_args = kwargs

        def disable_omni_model_cpu_offload(self) -> None:
            self.disable_called = True

    pipeline = CustomPipeline()
    backend = ModelLevelOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL, pin_cpu_memory=False),
        torch.device("cpu"),
    )

    backend.enable(pipeline)

    assert backend.enabled is True
    assert pipeline.enable_args == {
        "device": torch.device("cpu"),
        "pin_memory": False,
        "use_hsdp": False,
    }

    backend.disable()

    assert backend.enabled is False
    assert pipeline.disable_called is True


class TestMoveParamsPinMemory:
    def test_dtensor_skips_pin_memory(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """DTensor should skip pin_memory to avoid RuntimeError."""
        module = _create_simple_module().to(accelerator_device)
        tracker, mock_pin = _track_pin_memory_calls()

        original_isinstance = isinstance

        def fake_isinstance(obj, cls):
            if cls.__name__ == "DTensor":
                return True
            return original_isinstance(obj, cls)

        monkeypatch.setattr(torch.Tensor, "pin_memory", mock_pin)
        monkeypatch.setattr("builtins.isinstance", fake_isinstance)
        hook = SequentialOffloadHook(
            offload_targets=[],
            device=accelerator_device,
            pin_memory=True,
            use_hsdp=False,
        )
        hook._move_params(
            module,
            torch.device("cpu"),
            non_blocking=False,
            pin_memory=True,
        )
        assert not tracker["called"], "pin_memory should not be called for DTensor"

    def test_regular_tensor_calls_pin_memory(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """Regular tensor should call pin_memory when moving to CPU."""
        module = _create_simple_module().to(accelerator_device)
        tracker, mock_pin = _track_pin_memory_calls()

        monkeypatch.setattr(torch.Tensor, "pin_memory", mock_pin)
        hook = SequentialOffloadHook(
            offload_targets=[],
            device=accelerator_device,
            pin_memory=True,
            use_hsdp=False,
        )
        hook._move_params(
            module,
            torch.device("cpu"),
            non_blocking=False,
            pin_memory=True,
        )
        assert tracker["called"], "pin_memory should be called for regular tensors"

    def test_pin_memory_skipped_when_disabled(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """pin_memory should not be called when pin_memory=False."""
        module = _create_simple_module().to(accelerator_device)
        tracker, mock_pin = _track_pin_memory_calls()

        monkeypatch.setattr(torch.Tensor, "pin_memory", mock_pin)
        hook = SequentialOffloadHook(
            offload_targets=[],
            device=accelerator_device,
            pin_memory=False,
            use_hsdp=False,
        )
        hook._move_params(
            module,
            torch.device("cpu"),
            non_blocking=False,
            pin_memory=False,
        )
        assert not tracker["called"], "pin_memory should not be called when disabled"

    def test_pin_memory_skipped_for_non_cpu_target(self, accelerator_device, monkeypatch: pytest.MonkeyPatch):
        """pin_memory should not be called for non-CPU targets."""
        module = _create_simple_module().to("cpu")
        tracker, mock_pin = _track_pin_memory_calls()

        monkeypatch.setattr(torch.Tensor, "pin_memory", mock_pin)
        hook = SequentialOffloadHook(
            offload_targets=[],
            device=torch.device("cpu"),
            pin_memory=True,
            use_hsdp=False,
        )
        hook._move_params(module, accelerator_device, non_blocking=False, pin_memory=True)
        assert not tracker["called"], "pin_memory should not be called for non-CPU target"


class _StageableVAE(nn.Module):
    """A VAE that manages its own residency, like MiniMax-H3's adapters."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.moves: list[str] = []

    def load_to_device(self) -> None:
        self.moves.append("load")

    def offload_to_cpu(self) -> None:
        self.moves.append("offload")

    def to(self, *args, **kwargs):  # noqa: A003 - mirrors nn.Module.to
        self.moves.append("to")
        return super().to(*args, **kwargs)


class _PlainVAE(nn.Module):
    """A VAE without the staging pair: it cannot be parked in host memory."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.moves: list[str] = []

    def to(self, *args, **kwargs):  # noqa: A003 - mirrors nn.Module.to
        self.moves.append("to")
        return super().to(*args, **kwargs)


def _pipeline_with(vae: nn.Module, *, declared: bool = True, with_encoder: bool = True) -> nn.Module:
    from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

    class Pipeline(nn.Module):
        _dit_modules = ["transformer"]
        _encoder_modules = ["text_encoder"] if with_encoder else []
        _vae_modules = ["vae"]
        _offload_plan = OffloadPlan(on_demand_component_paths=frozenset({"vae"} if declared else set()))

        def __init__(self) -> None:
            super().__init__()
            self.transformer = _create_simple_module()
            if with_encoder:
                self.text_encoder = _create_simple_module()
            self.vae = vae

    return Pipeline()


def _model_level_backend(*, vae_cpu_offload: bool) -> ModelLevelOffloadBackend:
    return ModelLevelOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.MODEL_LEVEL,
            pin_cpu_memory=False,
            vae_cpu_offload=vae_cpu_offload,
        ),
        torch.device("cpu"),
    )


def test_model_level_backend_confirms_host_residency_for_declared_vae() -> None:
    """The declared, stageable VAE is parked, not moved to the device."""
    vae = _StageableVAE()
    backend = _model_level_backend(vae_cpu_offload=True)

    backend.enable(_pipeline_with(vae))

    # Idempotent confirmation, and never a move onto the device.
    assert vae.moves == ["offload"]
    backend.disable()


def test_model_level_backend_keeps_vae_resident_without_the_policy() -> None:
    """Default deployments keep the legacy placement."""
    vae = _StageableVAE()
    backend = _model_level_backend(vae_cpu_offload=False)

    backend.enable(_pipeline_with(vae))

    assert vae.moves == ["to"]
    backend.disable()


def test_model_level_backend_keeps_undeclared_vae_resident() -> None:
    """Exposing the staging pair is not enough: the pipeline must declare it."""
    vae = _StageableVAE()
    backend = _model_level_backend(vae_cpu_offload=True)

    backend.enable(_pipeline_with(vae, declared=False))

    assert vae.moves == ["to"]
    backend.disable()


def test_model_level_backend_keeps_unstageable_vae_resident() -> None:
    """A declared VAE that cannot stage itself must not be parked."""
    vae = _PlainVAE()
    backend = _model_level_backend(vae_cpu_offload=True)

    backend.enable(_pipeline_with(vae))

    assert vae.moves == ["to"]
    backend.disable()


def test_model_level_backend_parks_vae_even_when_no_encoder_is_found() -> None:
    """The VAE decision must land before the no-encoder early return."""
    vae = _StageableVAE()
    backend = _model_level_backend(vae_cpu_offload=True)

    backend.enable(_pipeline_with(vae, with_encoder=False))

    assert vae.moves == ["offload"]


def test_model_level_backend_enable_disable_enable_does_not_stack_hooks() -> None:
    vae = _StageableVAE()
    pipeline = _pipeline_with(vae)
    backend = _model_level_backend(vae_cpu_offload=True)

    def hook_count() -> int:
        total = 0
        for module in (pipeline.transformer, pipeline.text_encoder, pipeline.vae):
            registry = getattr(module, "_hook_registry", None)
            if registry is not None and SequentialOffloadHook._HOOK_NAME in registry._hooks:
                total += 1
        return total

    backend.enable(pipeline)
    first = hook_count()
    backend.disable()
    assert hook_count() == 0
    backend.enable(pipeline)
    assert hook_count() == first
    backend.disable()


def test_offload_config_carries_vae_cpu_offload_from_od_config() -> None:
    """The live consumer must see the operator's choice, not a default.

    ``OffloadConfig.from_od_config`` is the last hop before the backend reads
    the policy; every hop above it is silent on failure, so pin the value here.
    """
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

    def _od_config(**overrides):
        return OmniDiffusionConfig(
            model="MiniMaxAI/MiniMax-H3",
            enable_cpu_offload=True,
            parallel_config=DiffusionParallelConfig(cfg_parallel_size=1),
            **overrides,
        )

    config = OffloadConfig.from_od_config(_od_config(vae_cpu_offload=True))

    assert config.strategy is OffloadStrategy.MODEL_LEVEL
    assert config.vae_cpu_offload is True
    # Default stays off, so existing deployments keep the resident placement.
    assert OffloadConfig.from_od_config(_od_config()).vae_cpu_offload is False
