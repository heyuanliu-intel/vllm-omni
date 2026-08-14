# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for `GroupCoordinator.all_gather`'s reused landing buffer.

Communication libraries register the device memory handed to them and keep that
registration alive. Allocating a fresh output for every all-gather therefore
leaks registrations whenever the caching allocator hands back a different
address -- which any `empty_cache()` on the request path guarantees. The
coordinator now lands the collective in a reused buffer and copies out, so the
registered address is stable while callers keep owning their result.

That reuse is gated on XPU, because the copy-out is not free and the address
churn is specific to the XPU collective backend. These tests therefore drive
both halves of the gate: the reuse bodies force the platform to report XPU, and
`_body_non_xpu_path_unchanged` pins that every other platform still gets main's
allocate-and-gather path with no landing buffer and no copy.

These run on gloo/CPU (no accelerator), matching `test_comm.py`.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed import group_coordinator as gc_module
from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator


class _AsXPU:
    """`current_omni_platform` stand-in that only overrides `is_xpu`.

    The reuse path is XPU-gated, but nothing in it is device-specific -- CPU
    tensors over gloo exercise the same code -- so the gate is flipped rather
    than requiring an accelerator.
    """

    def __init__(self, real):
        self._real = real

    def is_xpu(self) -> bool:
        return True

    def __getattr__(self, name):
        return getattr(self._real, name)


def _run(
    body: Callable[[GroupCoordinator, int, int], None],
    local_rank: int,
    world_size: int,
    master_port: int,
    as_xpu: bool = True,
) -> None:
    """Build a gloo group and a coordinator directly.

    Deliberately *not* via `init_distributed_environment`: that helper does
    `get_rank() % get_device_count()` and therefore needs at least one visible
    accelerator, which an L1 CPU runner does not have. Everything under test
    here is device-agnostic -- CPU tensors over gloo exercise the same code
    path -- so the coordinator is constructed directly and stays runnable with
    zero accelerators.
    """
    for key, value in {
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": str(master_port),
    }.items():
        os.environ[key] = value

    dist.init_process_group(backend="gloo", init_method="env://", world_size=world_size, rank=local_rank)
    real_platform = gc_module.current_omni_platform
    if as_xpu:
        gc_module.current_omni_platform = _AsXPU(real_platform)
    try:
        group = GroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=local_rank,
            torch_distributed_backend="gloo",
        )
        body(group, local_rank, world_size)
    finally:
        gc_module.current_omni_platform = real_platform
        dist.destroy_process_group()


def _rank_tensor(local_rank: int, rows: int = 3, cols: int = 4) -> torch.Tensor:
    """Rank-distinguishable payload, so a wrong gather cannot look right."""
    return torch.full((rows, cols), float(local_rank + 1)) + torch.arange(rows * cols, dtype=torch.float32).reshape(
        rows, cols
    )


def _expected(world_size: int, rows: int = 3, cols: int = 4) -> torch.Tensor:
    return torch.cat([_rank_tensor(r, rows, cols) for r in range(world_size)], dim=0)


# ---------------------------------------------------------------------------
# worker bodies
# ---------------------------------------------------------------------------


def _body_parity(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """Reused buffer must produce exactly what a fresh allocation produces."""
    out = group.all_gather(_rank_tensor(local_rank))
    torch.testing.assert_close(out, _expected(world_size), rtol=0, atol=0)


def _body_landing_address_is_stable(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """The property that actually fixes the leak: one registered address.

    Three same-shape gathers must all land in the same buffer. Without reuse the
    coordinator allocates a new output each time and this dict stays empty.
    """
    seen = []
    for _ in range(3):
        group.all_gather(_rank_tensor(local_rank))
        buffers = group._all_gather_buffers
        assert buffers, "reuse is enabled but no landing buffer was cached"
        assert len(buffers) == 1, f"expected one buffer per (dtype, device), got {len(buffers)}"
        seen.append(next(iter(buffers.values())).data_ptr())
    assert len(set(seen)) == 1, f"landing buffer moved between calls: {seen}"


def _body_result_is_private(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """Callers still own their result -- the copy-out must not be skipped.

    A caller may hold the previous result across the next gather; if we handed
    back the landing buffer itself, the second gather would rewrite the first
    result in place.
    """
    first = group.all_gather(_rank_tensor(local_rank))
    first_snapshot = first.clone()
    first.mul_(-1.0)  # a caller mutating what it owns must not corrupt the buffer

    second = group.all_gather(_rank_tensor(local_rank))
    torch.testing.assert_close(second, _expected(world_size), rtol=0, atol=0)
    torch.testing.assert_close(first, -first_snapshot, rtol=0, atol=0)
    assert first.data_ptr() != second.data_ptr(), "two results must not alias each other"


def _body_non_zero_dim(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """`dim != 0` reshapes the gathered result; reuse must not disturb it."""
    out = group.all_gather(_rank_tensor(local_rank), dim=1)
    expected = torch.cat([_rank_tensor(r) for r in range(world_size)], dim=1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _body_separate_tensors(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """`separate_tensors=True` returns per-rank views; they must stay correct."""
    parts = group.all_gather(_rank_tensor(local_rank), separate_tensors=True)
    assert len(parts) == world_size
    for rank, part in enumerate(parts):
        torch.testing.assert_close(part, _rank_tensor(rank), rtol=0, atol=0)


def _body_grows_for_larger_shape(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """A bigger gather grows the single buffer instead of adding another."""
    group.all_gather(_rank_tensor(local_rank, rows=2))
    small = next(iter(group._all_gather_buffers.values())).numel()
    group.all_gather(_rank_tensor(local_rank, rows=8))
    buffers = group._all_gather_buffers
    assert len(buffers) == 1, f"buffer count must stay 1 per (dtype, device), got {len(buffers)}"
    grown = next(iter(buffers.values())).numel()
    assert grown > small, "buffer did not grow for the larger gather"
    # Going back below the record high must not shrink or re-key the buffer:
    # that is what bounds the residency by the largest gather instead of by the
    # number of shapes, and what keeps the address stable afterwards.
    group.all_gather(_rank_tensor(local_rank, rows=2))
    buffers = group._all_gather_buffers
    assert len(buffers) == 1, f"a smaller gather added a second buffer: {len(buffers)}"
    assert next(iter(buffers.values())).numel() == grown, "buffer shrank below its record high"


def _body_non_xpu_path_unchanged(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """Off XPU, `all_gather` must be exactly what main does.

    Two properties together pin that: no landing buffer is ever created, and the
    tensor handed back to the caller *is* the tensor the collective wrote into
    -- i.e. there is no extra clone on the hot path.
    """
    assert not gc_module.current_omni_platform.is_xpu(), "this body must run with the gate closed"

    real_all_gather_into_tensor = dist.all_gather_into_tensor
    written_to: list[int] = []

    def recording(output, input_, **kwargs):
        written_to.append(output.data_ptr())
        return real_all_gather_into_tensor(output, input_, **kwargs)

    dist.all_gather_into_tensor = recording
    try:
        for _ in range(3):
            out = group.all_gather(_rank_tensor(local_rank))
            torch.testing.assert_close(out, _expected(world_size), rtol=0, atol=0)
            assert out.data_ptr() == written_to[-1], "off XPU the result must not be a copy"
            assert group._all_gather_buffers is None, "off XPU no landing buffer may be allocated"
    finally:
        dist.all_gather_into_tensor = real_all_gather_into_tensor

    # Note deliberately *not* asserted: that the three outputs sit at three
    # distinct addresses. Off XPU each call does allocate afresh, but the
    # caching allocator is free to hand back the block the previous result
    # released, so identical addresses are legal and would make this flaky.


# ---------------------------------------------------------------------------
# spawn wrappers
# ---------------------------------------------------------------------------


def _body_pipeline_subclass_inherits_all_gather(group: GroupCoordinator, local_rank: int, world_size: int) -> None:
    """The PP coordinator inherits `all_gather` without calling `super().__init__`.

    `PipelineGroupCoordinator` re-implements `__init__`, so any per-instance
    state the base constructor sets up is absent there while the inherited
    `all_gather` still runs. Build one through the real factory and exercise the
    inherited method.
    """
    from vllm_omni.diffusion.distributed.parallel_state import init_model_parallel_group

    pp_group = init_model_parallel_group(
        group_ranks=[list(range(world_size))],
        local_rank=local_rank,
        backend="gloo",
        parallel_mode="pipeline",
    )
    out = pp_group.all_gather(_rank_tensor(local_rank))
    torch.testing.assert_close(out, _expected(world_size), rtol=0, atol=0)


_BODIES = {
    "parity": _body_parity,
    "pipeline": _body_pipeline_subclass_inherits_all_gather,
    "stable": _body_landing_address_is_stable,
    "private": _body_result_is_private,
    "dim1": _body_non_zero_dim,
    "separate": _body_separate_tensors,
    "grow": _body_grows_for_larger_shape,
    "non_xpu": _body_non_xpu_path_unchanged,
}


def _entry(local_rank: int, world_size: int, master_port: int, body_name: str, as_xpu: bool) -> None:
    _run(_BODIES[body_name], local_rank, world_size, master_port, as_xpu=as_xpu)


def _spawn(world_size: int, master_port: int, body_name: str, as_xpu: bool = True) -> None:
    torch.multiprocessing.spawn(
        _entry,
        args=(world_size, master_port, body_name, as_xpu),
        nprocs=world_size,
    )


# ---------------------------------------------------------------------------
# CPU: gloo collectives on CPU tensors (no accelerator)
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_matches_fresh_allocation(world_size: int):
    _spawn(world_size, 29660 + world_size, "parity")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_pipeline_coordinator_inherits_all_gather(world_size: int):
    """MRO regression: the subclass that skips `super().__init__` must still work."""
    _spawn(world_size, 29740 + world_size, "pipeline")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_landing_buffer_address_is_stable(world_size: int):
    _spawn(world_size, 29680 + world_size, "stable")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_result_does_not_alias_landing_buffer(world_size: int):
    _spawn(world_size, 29690 + world_size, "private")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_non_zero_dim(world_size: int):
    _spawn(world_size, 29700 + world_size, "dim1")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_separate_tensors(world_size: int):
    _spawn(world_size, 29710 + world_size, "separate")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_landing_buffer_grows_instead_of_multiplying(world_size: int):
    _spawn(world_size, 29720 + world_size, "grow")


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_non_xpu_all_gather_is_unchanged(world_size: int):
    """The other half of the gate: no landing buffer and no copy off XPU."""
    _spawn(world_size, 29750 + world_size, "non_xpu", as_xpu=False)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.cpu
@pytest.mark.parametrize("world_size", [2, 4])
def test_non_xpu_all_gather_matches_fresh_allocation(world_size: int):
    """Correctness parity on the ungated path, for completeness."""
    _spawn(world_size, 29760 + world_size, "parity", as_xpu=False)
