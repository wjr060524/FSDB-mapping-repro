"""Differentiable-shadow / hard-forward primitives for FSDB QAT.

The integer projection is deliberately separate from the inference-only PTQ
backend.  Every call re-encodes the current floating-point shadow weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .quantizer import (
    CANONICAL_FIELDS,
    LEGACY_CODEBOOK,
    fsdb_decode,
    legacy_integer_target,
)


@dataclass(frozen=True)
class QATHardProjection:
    q: torch.Tensor
    target: torch.Tensor
    hard_weight: torch.Tensor
    mag: torch.Tensor
    group_sign: torch.Tensor
    group_shift: torch.Tensor
    group_index: torch.Tensor
    ratio: int


def _as_weight_scalar(value, weight: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=weight.dtype, device=weight.device)


def _dequantize(q: torch.Tensor, weight: torch.Tensor, scale, zero_point) -> torch.Tensor:
    scale_t = _as_weight_scalar(scale, weight)
    zero_t = _as_weight_scalar(zero_point, weight)
    if float(scale_t.item()) == 0.0:
        raise ValueError("weight scale must be non-zero")
    return (q.to(weight.dtype) - zero_t) / scale_t


def _ratio_one(
    weight: torch.Tensor,
    scale,
    zero_point,
    qmin: int,
    qmax: int,
) -> QATHardProjection:
    target = legacy_integer_target(weight, scale, zero_point, qmin, qmax)
    # target is integral and bounded, so a 256-entry LUT is exactly equivalent
    # to materialising [..., 22] distances for every ResNet weight.
    target_values = torch.arange(qmin, qmax + 1, dtype=torch.int64, device=weight.device)
    codebook = torch.tensor(LEGACY_CODEBOOK, dtype=torch.int64, device=weight.device)
    nearest_index = (target_values[:, None] - codebook[None, :]).abs().argmin(dim=1)
    q_lut = codebook[nearest_index]
    lut_index = target - qmin
    q = q_lut[lut_index]
    canonical = [CANONICAL_FIELDS[int(code)] for code in LEGACY_CODEBOOK]
    mag_codes = torch.tensor([item[0] for item in canonical], dtype=torch.uint8, device=weight.device)
    sign_codes = torch.tensor([item[1] for item in canonical], dtype=torch.uint8, device=weight.device)
    shift_codes = torch.tensor([item[2] for item in canonical], dtype=torch.uint8, device=weight.device)
    mag = mag_codes[nearest_index][lut_index]
    sign = sign_codes[nearest_index][lut_index]
    shift = shift_codes[nearest_index][lut_index]
    group_index = torch.arange(q.numel(), dtype=torch.int64, device=q.device).reshape(q.shape)
    return QATHardProjection(
        q=q,
        target=target,
        hard_weight=_dequantize(q, weight, scale, zero_point),
        mag=mag,
        group_sign=sign.reshape(-1),
        group_shift=shift.reshape(-1),
        group_index=group_index,
        ratio=1,
    )


def _ratio_grouped(
    weight: torch.Tensor,
    scale,
    zero_point,
    qmin: int,
    qmax: int,
    ratio: int,
) -> QATHardProjection:
    if weight.ndim < 1:
        raise ValueError("weight must have an output-channel dimension")
    # The required layout is independent grouping inside every output channel.
    output_channels = weight.shape[0]
    per_channel = weight.numel() // output_channels
    groups_per_channel = (per_channel + ratio - 1) // ratio
    padded = groups_per_channel * ratio

    compute_dtype = weight.dtype if weight.is_floating_point() else torch.float32
    scale_t = torch.as_tensor(scale, dtype=compute_dtype, device=weight.device)
    zero_t = torch.as_tensor(zero_point, dtype=compute_dtype, device=weight.device)
    target = torch.clamp(
        torch.round(weight.to(compute_dtype) * scale_t + zero_t), qmin, qmax
    ).to(torch.int64)
    target_flat = target.reshape(output_channels, per_channel)
    padded_target = torch.zeros(
        (output_channels, padded), dtype=torch.int64, device=weight.device
    )
    padded_target[:, :per_channel] = target_flat
    grouped_target = padded_target.reshape(output_channels, groups_per_channel, ratio)
    valid = torch.arange(padded, device=weight.device).reshape(groups_per_channel, ratio) < per_channel
    valid = valid.unsqueeze(0)

    # Stable tie order: (sign, shift) = 00, 01, 10, 11; mag = 0..7.
    states: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
    state_errors = []
    state_q = []
    state_mag = []
    mags = torch.arange(8, dtype=torch.int64, device=weight.device)
    for sign, shift in states:
        candidates = fsdb_decode(
            mags,
            torch.full_like(mags, sign),
            torch.full_like(mags, shift),
        )
        target_values = torch.arange(qmin, qmax + 1, dtype=torch.int64, device=weight.device)
        lut_mag = (target_values[:, None] - candidates[None, :]).abs().argmin(dim=-1)
        mag_choice = lut_mag[grouped_target - qmin]
        q_choice = candidates[mag_choice]
        error = ((q_choice - grouped_target).to(torch.float64).square() * valid).sum(dim=-1)
        state_errors.append(error)
        state_q.append(q_choice)
        state_mag.append(mag_choice)

    winning_state = torch.stack(state_errors, dim=-1).argmin(dim=-1)
    q_states = torch.stack(state_q, dim=-1)
    mag_states = torch.stack(state_mag, dim=-1)
    gather_index = winning_state.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ratio, 1)
    grouped_q = torch.gather(q_states, -1, gather_index).squeeze(-1)
    grouped_mag = torch.gather(mag_states, -1, gather_index).squeeze(-1).to(torch.uint8)

    q = grouped_q.reshape(output_channels, padded)[:, :per_channel].reshape(weight.shape)
    mag = grouped_mag.reshape(output_channels, padded)[:, :per_channel].reshape(weight.shape)
    sign_lut = torch.tensor([s for s, _ in states], dtype=torch.uint8, device=weight.device)
    shift_lut = torch.tensor([f for _, f in states], dtype=torch.uint8, device=weight.device)
    group_sign = sign_lut[winning_state].reshape(-1)
    group_shift = shift_lut[winning_state].reshape(-1)
    group_ids = torch.arange(
        output_channels * groups_per_channel, dtype=torch.int64, device=weight.device
    ).reshape(output_channels, groups_per_channel, 1).expand(-1, -1, ratio)
    group_index = group_ids.reshape(output_channels, padded)[:, :per_channel].reshape(weight.shape)

    decoded = fsdb_decode(mag, group_sign[group_index], group_shift[group_index])
    if not torch.equal(decoded, q):
        raise RuntimeError("FSDB grouped encode/decode round trip failed")
    return QATHardProjection(
        q=q,
        target=target,
        hard_weight=_dequantize(q, weight, scale, zero_point),
        mag=mag,
        group_sign=group_sign,
        group_shift=group_shift,
        group_index=group_index,
        ratio=ratio,
    )


def project_qat_weight(
    shadow_weight: torch.Tensor,
    scale,
    zero_point=0,
    qmin: int = -128,
    qmax: int = 127,
    ratio: int = 1,
) -> QATHardProjection:
    """Project a shadow tensor to the legal hard FSDB surface."""
    ratio = int(ratio)
    if ratio == 1:
        return _ratio_one(shadow_weight, scale, zero_point, qmin, qmax)
    if ratio in (4, 16):
        return _ratio_grouped(shadow_weight, scale, zero_point, qmin, qmax, ratio)
    raise ValueError("FSDB QAT currently supports only ratio=1, ratio=4, or ratio=16")


def project_qat_weight_fixed_group_control(
    shadow_weight: torch.Tensor,
    scale,
    zero_point,
    qmin: int,
    qmax: int,
    ratio: int,
    group_sign: torch.Tensor,
    group_shift: torch.Tensor,
) -> QATHardProjection:
    """Project magnitudes while preserving externally supplied group controls.

    This is intentionally a separate entry point so the default dynamic QAT
    projection remains unchanged.  Magnitudes are reselected independently for
    the current shadow weights; only the shared sign/shift state is fixed.
    """
    ratio = int(ratio)
    if ratio not in (4, 16):
        raise ValueError("fixed group control requires ratio=4 or ratio=16")
    if shadow_weight.ndim < 1:
        raise ValueError("weight must have an output-channel dimension")

    output_channels = shadow_weight.shape[0]
    per_channel = shadow_weight.numel() // output_channels
    groups_per_channel = (per_channel + ratio - 1) // ratio
    padded = groups_per_channel * ratio
    group_count = output_channels * groups_per_channel

    sign = group_sign.to(device=shadow_weight.device, dtype=torch.uint8).reshape(-1)
    shift = group_shift.to(device=shadow_weight.device, dtype=torch.uint8).reshape(-1)
    if sign.numel() != group_count or shift.numel() != group_count:
        raise ValueError(
            f"fixed group-control size mismatch: expected {group_count}, "
            f"got sign={sign.numel()} shift={shift.numel()}"
        )
    if not bool(((sign <= 1) & (shift <= 1)).all()):
        raise ValueError("fixed group sign/shift must be binary")

    compute_dtype = shadow_weight.dtype if shadow_weight.is_floating_point() else torch.float32
    scale_t = torch.as_tensor(scale, dtype=compute_dtype, device=shadow_weight.device)
    zero_t = torch.as_tensor(zero_point, dtype=compute_dtype, device=shadow_weight.device)
    target = torch.clamp(
        torch.round(shadow_weight.to(compute_dtype) * scale_t + zero_t), qmin, qmax
    ).to(torch.int64)

    group_ids = torch.arange(group_count, dtype=torch.int64, device=shadow_weight.device)
    group_index = group_ids.reshape(output_channels, groups_per_channel, 1).expand(
        -1, -1, ratio
    ).reshape(output_channels, padded)[:, :per_channel].reshape(shadow_weight.shape)
    position_state = (
        sign[group_index].to(torch.int64) * 2 + shift[group_index].to(torch.int64)
    )

    q = torch.empty_like(target)
    mag = torch.empty_like(target, dtype=torch.uint8)
    mags = torch.arange(8, dtype=torch.int64, device=shadow_weight.device)
    target_values = torch.arange(qmin, qmax + 1, dtype=torch.int64, device=shadow_weight.device)
    states: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))
    for state_index, (state_sign, state_shift) in enumerate(states):
        candidates = fsdb_decode(
            mags,
            torch.full_like(mags, state_sign),
            torch.full_like(mags, state_shift),
        )
        lut_mag = (target_values[:, None] - candidates[None, :]).abs().argmin(dim=-1)
        selected_mag = lut_mag[target - qmin]
        mask = position_state == state_index
        mag[mask] = selected_mag[mask].to(torch.uint8)
        q[mask] = candidates[selected_mag[mask]]

    decoded = fsdb_decode(mag, sign[group_index], shift[group_index])
    if not torch.equal(decoded, q):
        raise RuntimeError("fixed-control FSDB encode/decode round trip failed")
    return QATHardProjection(
        q=q,
        target=target,
        hard_weight=_dequantize(q, shadow_weight, scale, zero_point),
        mag=mag,
        group_sign=sign,
        group_shift=shift,
        group_index=group_index,
        ratio=ratio,
    )
