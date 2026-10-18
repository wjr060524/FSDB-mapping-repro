"""Authoritative integer primitives for the FSDB PTQ backend."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


LEGACY_CODEBOOK = (
    -64, -56, -48, -40, -32, -24, -16, -8,
    -4, -3, -1, 0, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56,
)


@dataclass(frozen=True)
class FSDBEncoding:
    mag: torch.Tensor
    sign: torch.Tensor
    shift: torch.Tensor
    q: torch.Tensor
    target: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int


def fsdb_decode(mag: torch.Tensor, sign: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Decode integer FSDB fields using the paper formula supplied for this task."""
    mag_i = mag.to(torch.int64)
    sign_i = sign.to(torch.int64)
    shift_i = shift.to(torch.int64)
    if torch.any((mag_i < 0) | (mag_i > 7)):
        raise ValueError("mag must be representable by 3 bits (0..7)")
    if torch.any((sign_i < 0) | (sign_i > 1)):
        raise ValueError("sign must be 0 or 1")
    if torch.any((shift_i < 0) | (shift_i > 1)):
        raise ValueError("shift must be 0 or 1")
    low = -4 * sign_i + mag_i
    high = 8 * (-8 * sign_i + mag_i)
    return torch.where(shift_i == 0, low, high)


def legacy_codebook_to_canonical_fields() -> Dict[int, Tuple[int, int, int]]:
    """Return deterministic q -> (mag, sign, shift) mappings.

    Ties are ordered by shift, then magnitude, then sign as requested.
    """
    result: Dict[int, Tuple[int, int, int]] = {}
    for q in LEGACY_CODEBOOK:
        candidates = []
        for shift in (0, 1):
            for mag in range(8):
                for sign in (0, 1):
                    decoded = int(fsdb_decode(
                        torch.tensor(mag), torch.tensor(sign), torch.tensor(shift)
                    ).item())
                    if decoded == q:
                        candidates.append((shift, mag, sign))
        if not candidates:
            raise RuntimeError(f"legacy code {q} is not representable by FSDB fields")
        shift, mag, sign = min(candidates)
        result[q] = (mag, sign, shift)
    return result


CANONICAL_FIELDS = legacy_codebook_to_canonical_fields()


def compute_legacy_weight_scale(weight: torch.Tensor, wbit: int) -> torch.Tensor:
    if not 2 <= int(wbit) <= 16:
        raise ValueError("wbit must be between 2 and 16")
    max_abs = weight.detach().abs().max().to(torch.float64)
    if max_abs.item() == 0:
        return torch.ones((), dtype=torch.float64, device=weight.device)
    qmin = -(2 ** (int(wbit) - 1))
    qmax = 2 ** (int(wbit) - 1) - 1
    return torch.tensor((qmax - qmin) / 2, dtype=torch.float64, device=weight.device) / max_abs


def legacy_integer_target(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | float | int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """The rounded/clamped integer target shared by all ratios."""
    # Legacy QuantWeight.forward performs arithmetic in the weight tensor dtype.
    # Promoting float32 weights to float64 changes a small number of half-integer
    # rounding decisions and breaks exact 1:1 compatibility.
    compute_dtype = weight.dtype if weight.is_floating_point() else torch.float32
    scale_t = torch.as_tensor(scale, dtype=compute_dtype, device=weight.device)
    zero_t = torch.as_tensor(zero_point, dtype=compute_dtype, device=weight.device)
    return torch.clamp(torch.round(weight.to(compute_dtype) * scale_t + zero_t), qmin, qmax).to(torch.int64)


def legacy_project_codes(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | float | int,
    qmin: int,
    qmax: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project exactly as legacy QuantWeight.forward: round, clamp, nearest codebook."""
    target = legacy_integer_target(weight, scale, zero_point, qmin, qmax)
    codebook = torch.tensor(LEGACY_CODEBOOK, dtype=torch.int64, device=weight.device)
    distances = (target.unsqueeze(-1) - codebook).abs()
    q = codebook[distances.argmin(dim=-1)]
    return q, target


def _codes_to_fields(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mag = torch.empty_like(q, dtype=torch.uint8)
    sign = torch.empty_like(q, dtype=torch.uint8)
    shift = torch.empty_like(q, dtype=torch.uint8)
    assigned = torch.zeros_like(q, dtype=torch.bool)
    for code, (m, s, f) in CANONICAL_FIELDS.items():
        mask = q == code
        mag[mask] = m
        sign[mask] = s
        shift[mask] = f
        assigned |= mask
    if not bool(assigned.all()):
        unknown = torch.unique(q[~assigned]).detach().cpu().tolist()
        raise ValueError(f"codes outside legacy codebook: {unknown}")
    return mag, sign, shift


def fsdb_encode_1to1(
    weight: torch.Tensor,
    wbit: int = 8,
    scale: torch.Tensor | None = None,
    zero_point: float | int | torch.Tensor = 0,
) -> FSDBEncoding:
    """Create canonical 1:1 fields while preserving the legacy integer projection."""
    qmin = -(2 ** (int(wbit) - 1))
    qmax = 2 ** (int(wbit) - 1) - 1
    scale_t = compute_legacy_weight_scale(weight, wbit) if scale is None else torch.as_tensor(
        scale, dtype=torch.float64, device=weight.device
    )
    zero_t = torch.as_tensor(zero_point, dtype=torch.float64, device=weight.device)
    q, target = legacy_project_codes(weight, scale_t, zero_t, qmin, qmax)
    mag, sign, shift = _codes_to_fields(q)
    decoded = fsdb_decode(mag, sign, shift)
    if not torch.equal(decoded, q):
        mismatch = torch.nonzero(decoded != q, as_tuple=False)[0].tolist()
        raise RuntimeError(f"1:1 FSDB round-trip mismatch at index {mismatch}")
    return FSDBEncoding(mag, sign, shift, q, target, scale_t, zero_t, qmin, qmax)
