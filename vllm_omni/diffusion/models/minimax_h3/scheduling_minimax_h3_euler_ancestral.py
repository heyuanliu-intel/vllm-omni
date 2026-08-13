# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import torch


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_unit_timestep(timestep: torch.Tensor, name: str) -> None:
    if not isinstance(timestep, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(timestep):
        raise ValueError(f"{name} must be a floating point tensor")
    _require_finite_tensor(timestep, name)
    out_of_range = (timestep < 0) | (timestep > 1)
    if bool(out_of_range.any().item()):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"{name} must be finite")
    if sigma < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return sigma


def minimax_h3_rf_v_to_x0(
    xt: torch.Tensor,
    v: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if xt.shape != v.shape:
        raise ValueError(f"xt and v shapes must match, got {xt.shape} vs {v.shape}")
    if not torch.is_floating_point(xt):
        raise ValueError("xt must be a floating point tensor")
    if not torch.is_floating_point(v):
        raise ValueError("v must be a floating point tensor")
    _require_finite_tensor(xt, "xt")
    _require_finite_tensor(v, "v")
    _validate_unit_timestep(timestep, "timestep")
    cond_t = timestep.to(device=xt.device, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    sigma_t = 1 - cond_t
    x0 = xt + sigma_t * v
    _require_finite_tensor(x0, "x0")
    return x0


def minimax_h3_euler_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    if state.shape != denoised.shape:
        raise ValueError(f"state and denoised shapes must match, got {state.shape} vs {denoised.shape}")
    if not torch.is_floating_point(state):
        raise ValueError("state must be a floating point tensor")
    if not torch.is_floating_point(denoised):
        raise ValueError("denoised must be a floating point tensor")
    _require_finite_tensor(state, "state")
    _require_finite_tensor(denoised, "denoised")
    sigma_curr = _validate_sigma(sigma_curr, "sigma_curr")
    sigma_next = _validate_sigma(sigma_next, "sigma_next")
    if sigma_curr == 0.0:
        if sigma_next != 0.0:
            raise ValueError("sigma_next must be 0 when sigma_curr is 0")
        return state
    compute_dtype = torch.float32
    if state.dtype not in (torch.float16, torch.bfloat16):
        compute_dtype = state.dtype
    sigma_curr_t = state.new_tensor(sigma_curr, dtype=compute_dtype)
    sigma_next_t = state.new_tensor(sigma_next, dtype=compute_dtype)
    sigma_ratio = sigma_next_t / sigma_curr_t
    out = sigma_ratio * state.to(dtype=compute_dtype) + (1.0 - sigma_ratio) * denoised.to(dtype=compute_dtype)
    out = out.to(dtype=state.dtype)
    _require_finite_tensor(out, "euler_eta0_step output")
    return out


__all__ = [
    "minimax_h3_euler_eta0_step",
    "minimax_h3_rf_v_to_x0",
]
