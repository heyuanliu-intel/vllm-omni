# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The DiT must leave the card before a VAE stage runs, and the decoded
outputs must leave it before the DiT can come back.

Model-level CPU offload swaps the DiT against the encoders only, so the DiT is
still resident when decoding starts right after denoising and the VAE has to
allocate its output on top of it. The fix evicts every hooked DiT through
``evict_module_to_host`` and never restores inside the request: the sequential
hook's ``pre_forward`` reloads lazily. That reload happens on the *next seed's*
denoise when ``num_outputs_per_prompt > 1``, so ``decode`` also hands each
output to ``_offload_stage_output`` before returning -- otherwise the reload
would overlap with a resident output and recreate the original OOM.
"""

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline_module
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.offloader.sequential_backend import (
    SequentialOffloadHook,
    evict_module_to_host,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class _Recorder:
    """A pipeline stub carrying only what the eviction and decode touch."""

    def __init__(self, *, enable_cpu_offload: bool, layerwise: bool = False, dits: int = 1):
        self.calls: list[str] = []
        self.device = torch.device("cpu")
        # The real config class, so a renamed field breaks here rather than
        # silently defaulting inside the eviction's getattr chain.
        self.od_config = OmniDiffusionConfig(
            enable_cpu_offload=enable_cpu_offload,
            enable_layerwise_offload=layerwise,
        )
        self._dit_modules = [f"dit{i}" for i in range(dits)]
        for name in self._dit_modules:
            setattr(self, name, nn.Linear(2, 2))

    # Borrow the real predicates so the tests exercise production logic rather
    # than a second copy of it.
    _uses_manual_component_offload = MiniMaxH3Pipeline._uses_manual_component_offload
    _vae_stage_offload_active = MiniMaxH3Pipeline._vae_stage_offload_active


@pytest.fixture()
def evictions(monkeypatch: pytest.MonkeyPatch) -> list[nn.Module]:
    """Record which modules production hands to evict_module_to_host."""
    recorded: list[nn.Module] = []

    def _record(module: nn.Module) -> bool:
        recorded.append(module)
        return True

    monkeypatch.setattr(pipeline_module, "evict_module_to_host", _record)
    return recorded


def test_evicts_each_dit(evictions: list[nn.Module]) -> None:
    rec = _Recorder(enable_cpu_offload=True, dits=2)

    MiniMaxH3Pipeline._evict_resident_dits_for_vae(rec)

    assert evictions == [rec.dit0, rec.dit1], "every DiT must be handed to evict_module_to_host, in order"


def test_inert_without_cpu_offload(evictions: list[nn.Module]) -> None:
    rec = _Recorder(enable_cpu_offload=False)

    MiniMaxH3Pipeline._evict_resident_dits_for_vae(rec)

    assert evictions == [], "nothing may move when the operator did not enable CPU offload"


def test_inert_under_layerwise_offload(evictions: list[nn.Module]) -> None:
    # Layer-wise offload already streams the DiT block by block; evicting the
    # whole module on top of that would fight its own hooks.
    rec = _Recorder(enable_cpu_offload=True, layerwise=True)

    MiniMaxH3Pipeline._evict_resident_dits_for_vae(rec)

    assert evictions == []


def test_evict_module_to_host_refuses_unhooked_module() -> None:
    # No hook means no pre_forward to bring the module back later: evicting it
    # would strand it on the host.
    assert evict_module_to_host(nn.Linear(2, 2)) is False


def test_evict_module_to_host_accepts_hooked_module_already_home() -> None:
    dit = nn.Linear(2, 2)
    registry = HookRegistry.get_or_create(dit)
    registry.register_hook(
        SequentialOffloadHook._HOOK_NAME,
        SequentialOffloadHook(offload_targets=[], device=torch.device("cpu")),
    )

    assert evict_module_to_host(dit) is True, "a hooked module already on the host is a satisfied eviction"


class _StubVAE(nn.Module):
    def __init__(self, out: torch.Tensor):
        super().__init__()
        self._out = out

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self._out


def _decode_recorder(*, enable_cpu_offload: bool) -> tuple[_Recorder, torch.Tensor, torch.Tensor]:
    rec = _Recorder(enable_cpu_offload=enable_cpu_offload)
    rec.video_vae = _StubVAE(torch.zeros(1, 3, 2, 4, 4))
    rec.audio_vae = _StubVAE(torch.zeros(1, 2, 8))
    rec._component_on_device = MiniMaxH3Pipeline._component_on_device.__get__(rec)

    def _offload(tensor: torch.Tensor) -> torch.Tensor:
        rec.calls.append("offload_output")
        return tensor

    rec._offload_stage_output = _offload

    def _evict() -> None:
        rec.calls.append("evict")

    rec._evict_resident_dits_for_vae = _evict
    video, audio = MiniMaxH3Pipeline.decode(rec, torch.zeros(1), torch.zeros(1), height=4, width=4)
    return rec, video, audio


def test_decode_evicts_first_and_offloads_every_output() -> None:
    # One eviction ahead of both VAEs (per-component would pay the DiT D2H
    # twice), and both outputs leave the device before decode returns -- for
    # num_outputs_per_prompt > 1 the next thing that happens is the next
    # seed's denoise reloading the DiT.
    rec, _video, _audio = _decode_recorder(enable_cpu_offload=True)

    assert rec.calls == ["evict", "offload_output", "offload_output"], (
        "decode must evict once, before the video VAE, then hand video and audio to the stage-output offload"
    )


def test_offload_stage_output_is_identity_when_inactive() -> None:
    rec = _Recorder(enable_cpu_offload=False)
    tensor = torch.zeros(2)

    assert MiniMaxH3Pipeline._offload_stage_output(rec, tensor) is tensor, (
        "without CPU offload the decoded output must pass through untouched"
    )


def test_offload_stage_output_owner_rank_copies_to_host() -> None:
    rec = _Recorder(enable_cpu_offload=True)
    rec._is_output_owner_rank = lambda: True
    tensor = torch.zeros(2, 3)

    out = MiniMaxH3Pipeline._offload_stage_output(rec, tensor)

    assert out.device.type == "cpu" and out.shape == tensor.shape


def test_offload_stage_output_nonowner_rank_drops_storage_without_d2h() -> None:
    # Every rank runs the pipeline but the executor replies from rank 0 only;
    # a host copy on the other ranks would multiply host RAM by the world size
    # for outputs nobody consumes. They must keep only a shape-preserving
    # placeholder -- and it still has to survive the seed loop's torch.cat.
    rec = _Recorder(enable_cpu_offload=True)
    rec._is_output_owner_rank = lambda: False
    tensor = torch.zeros(2, 3, dtype=torch.float16)

    out = MiniMaxH3Pipeline._offload_stage_output(rec, tensor)

    assert out.device.type == "meta", "non-reply ranks must not pay a host copy"
    assert out.shape == tensor.shape and out.dtype == tensor.dtype
    assert torch.cat([out, out], dim=0).shape[0] == 4, "the placeholder must still concatenate"


def test_gate_ignores_widened_manual_predicate() -> None:
    # The gate must consult exactly the two layerwise flags. On the frozen base
    # _uses_manual_component_offload is equivalent, but contributors to that
    # predicate exist (host-staged VAEs: enable_cpu_offload AND vae_cpu_offload)
    # and they do NOT stream the DiT -- routing the gate through the predicate
    # silently disabled the eviction on exactly the target configuration.
    rec = _Recorder(enable_cpu_offload=True)
    rec._uses_manual_component_offload = lambda: True

    assert MiniMaxH3Pipeline._vae_stage_offload_active(rec) is True, (
        "a widened manual-staging predicate must not disable the DiT eviction"
    )


class _FakeDist:
    def __init__(self, *, available: bool, initialized: bool, rank: int):
        self._available = available
        self._initialized = initialized
        self._rank = rank

    def is_available(self) -> bool:
        return self._available

    def is_initialized(self) -> bool:
        return self._initialized

    def get_rank(self) -> int:
        return self._rank


@pytest.mark.parametrize(
    ("dist_state", "expected"),
    [
        (_FakeDist(available=False, initialized=False, rank=0), True),
        (_FakeDist(available=True, initialized=False, rank=0), True),
        (_FakeDist(available=True, initialized=True, rank=0), True),
        (_FakeDist(available=True, initialized=True, rank=1), False),
    ],
)
def test_is_output_owner_rank_follows_reply_contract(
    monkeypatch: pytest.MonkeyPatch, dist_state: _FakeDist, expected: bool
) -> None:
    # The executor replies from unique_reply_rank=0; the production predicate
    # (not a test stand-in) must say exactly that.
    monkeypatch.setattr(pipeline_module, "dist", dist_state)

    assert MiniMaxH3Pipeline._is_output_owner_rank() is expected


class _CountingTensor:
    """Stands in for a device tensor: the owner branch must call .cpu() once."""

    def __init__(self) -> None:
        self.cpu_calls = 0
        self.host = object()

    def cpu(self):
        self.cpu_calls += 1
        return self.host


def test_offload_stage_output_owner_pays_exactly_one_d2h() -> None:
    rec = _Recorder(enable_cpu_offload=True)
    rec._is_output_owner_rank = lambda: True
    tensor = _CountingTensor()

    out = MiniMaxH3Pipeline._offload_stage_output(rec, tensor)

    assert tensor.cpu_calls == 1 and out is tensor.host, (
        "the owner rank must actually move the output to the host, once"
    )


class _LogRecorder:
    def __init__(self) -> None:
        self.infos: list[tuple] = []
        self.warnings: list[tuple] = []

    def info(self, msg, *args) -> None:
        self.infos.append((msg, args))

    def warning(self, msg, *args) -> None:
        self.warnings.append((msg, args))

    def debug(self, msg, *args) -> None:
        pass


class _ResidentParam:
    device = torch.device("meta")

    @staticmethod
    def numel() -> int:
        return 1024

    @staticmethod
    def element_size() -> int:
        return 2


class _ResidentDiT:
    @staticmethod
    def parameters():
        return iter([_ResidentParam()])


def test_eviction_reports_moved_bytes(evictions: list, monkeypatch: pytest.MonkeyPatch) -> None:
    # The INFO line is the decision-point self-evidence that distinguishes a
    # real eviction from a silent no-op on a live serve log; it must carry the
    # parameter footprint that was actually resident.
    log = _LogRecorder()
    monkeypatch.setattr(pipeline_module, "logger", log)
    rec = _Recorder(enable_cpu_offload=True)
    rec.dit0 = _ResidentDiT()

    MiniMaxH3Pipeline._evict_resident_dits_for_vae(rec)

    assert len(log.infos) == 1 and log.infos[0][1][0] == 1, "one resident DiT evicted must be reported"
    assert log.infos[0][1][1] == pytest.approx(2048 / 1024**3), "the reported size is the resident parameter bytes"


def test_unhooked_resident_dit_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    # The dangerous silent path: resident DiT, no hook, nothing moves -- the
    # decode may OOM exactly as without the fix, and the log must say so.
    log = _LogRecorder()
    monkeypatch.setattr(pipeline_module, "logger", log)
    monkeypatch.setattr(pipeline_module, "evict_module_to_host", lambda module: False)
    rec = _Recorder(enable_cpu_offload=True)
    rec.dit0 = _ResidentDiT()

    MiniMaxH3Pipeline._evict_resident_dits_for_vae(rec)

    assert len(log.warnings) == 1 and log.warnings[0][1][0] == "dit0"
    assert log.infos == []


def _accelerator() -> torch.device | None:
    if torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return None


@hardware_test(res={"xpu": "B60", "cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(_accelerator() is None, reason="needs a real device: meta tensors cannot round-trip via .data")
def test_evict_module_to_host_moves_parameters_and_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    # The eviction and the hook's later restore both ride on _move_params, and
    # the DiT's rotary inv_freq lives in a buffer: a move that only covers
    # parameters produces wrong embeddings rather than a clean failure.
    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2)
            self.register_buffer("inv_freq", torch.ones(2))

    device = _accelerator()
    block = _Block().to(device)
    registry = HookRegistry.get_or_create(block)
    registry.register_hook(
        SequentialOffloadHook._HOOK_NAME,
        SequentialOffloadHook(offload_targets=[], device=device),
    )

    # A mid-stage eviction frees blocks so the same request's next allocation
    # can reuse them from the caching allocator; on XPU the hook's own
    # empty_cache is additionally a known source of collective receive-buffer
    # address churn. Zero calls is a load-bearing contract of this entry point,
    # so a refactor back onto hook._to_cpu must turn this red.
    from vllm_omni.platforms import current_omni_platform

    ec_calls = []
    monkeypatch.setattr(
        type(current_omni_platform), "empty_cache", classmethod(lambda cls: ec_calls.append(1))
    )

    assert evict_module_to_host(block) is True
    assert ec_calls == [], "evict_module_to_host must never call empty_cache"
    host = torch.device("cpu")
    assert all(p.device == host for p in block.parameters()), "parameters must move to the host"
    assert all(b.device == host for b in block.buffers()), "buffers must move with the parameters"