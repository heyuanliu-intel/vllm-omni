# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for `GroupCoordinator.all_gather`'s reused landing buffer.

Communication libraries register the device memory handed to them and keep that
registration alive. Allocating a fresh output for every all-gather therefore
leaks registrations whenever the caching allocator hands back a different
address -- which any `empty_cache()` on the request path guarantees. The
coordinator now lands the collective in a reused buffer and copies out, so the
registered address is stable while callers keep owning their result.

These run on gloo/CPU (no accelerator), matching `test_comm.py`.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator


def _run(
    body: Callable[[GroupCoordinator, int, int], None], local_rank: int, world_size: int, master_port: int
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
    try:
        group = GroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=local_rank,
            torch_distributed_backend="gloo",
        )
        body(group, local_rank, world_size)
    finally:
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
}


def _entry(local_rank: int, world_size: int, master_port: int, body_name: str) -> None:
    _run(_BODIES[body_name], local_rank, world_size, master_port)


def _spawn(world_size: int, master_port: int, body_name: str) -> None:
    torch.multiprocessing.spawn(
        _entry,
        args=(world_size, master_port, body_name),
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
