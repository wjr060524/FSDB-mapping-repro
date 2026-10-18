"""Mapping-aware FSDB QAT with immutable ROM magnitudes.

This module is intentionally isolated from the legacy contiguous QAT path.
Group membership is supplied as logical flattened convolution coordinates and
stored as a persistent buffer.  Shadow weights receive an identity STE while
the convolution always uses the exactly decoded hard FSDB weights.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .qat_frozen_mag import FrozenMagQATConv2dWrapper, FrozenMagQATModelWrapper
from .qat_quantizer import QATHardProjection
from .qat_wrapper import R18_LEGACY16_LAYERS
from .quantizer import fsdb_decode


def _validated_members(group_members: torch.Tensor, numel: int, ratio: int) -> torch.Tensor:
    members = torch.as_tensor(group_members, dtype=torch.int64).detach().clone()
    if members.ndim != 2 or members.shape[1] != ratio:
        raise ValueError(f"group_members must have shape [groups,{ratio}]")
    if bool((members < -1).any()) or bool((members >= numel).any()):
        raise ValueError("group_members contains an invalid logical coordinate")
    valid = members >= 0
    seen = members[valid]
    if seen.numel() != numel or torch.unique(seen).numel() != numel:
        raise ValueError("group_members must cover every logical coordinate exactly once")
    # Padding is legal only as a suffix of a row.
    if members.shape[1] > 1 and bool(((~valid[:, :-1]) & valid[:, 1:]).any()):
        raise ValueError("group_members padding must be a row suffix")
    return members


def _mapping_index(
    group_members: torch.Tensor, numel: int, weight_shape: torch.Size
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = group_members >= 0
    safe = group_members.clamp_min(0)
    flat_group_index = torch.empty(numel, dtype=torch.int64, device=group_members.device)
    group_ids = torch.arange(group_members.shape[0], dtype=torch.int64, device=group_members.device)
    expanded = group_ids[:, None].expand_as(group_members)
    flat_group_index[safe[valid]] = expanded[valid]
    return valid, safe, flat_group_index.reshape(weight_shape)


def project_mapping_fixed_mag(
    shadow_weight: torch.Tensor,
    frozen_mag: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_members: torch.Tensor,
    ratio: int,
    qmin: int = -128,
    qmax: int = 127,
    fixed_group_sign: torch.Tensor | None = None,
    fixed_group_shift: torch.Tensor | None = None,
) -> QATHardProjection:
    """Project using fixed logical groups and fixed per-weight magnitudes."""
    ratio = int(ratio)
    if ratio not in (4, 16):
        raise ValueError("mapping-aware projection requires ratio 4 or 16")
    if shadow_weight.shape != frozen_mag.shape:
        raise ValueError("frozen_mag shape must equal shadow_weight shape")
    if group_members.ndim != 2 or tuple(group_members.shape)[1] != ratio:
        raise ValueError("group_members ratio mismatch")
    mag = frozen_mag.to(device=shadow_weight.device, dtype=torch.uint8)
    members = group_members.to(device=shadow_weight.device, dtype=torch.int64)
    if bool((mag > 7).any()):
        raise ValueError("frozen_mag must be in 0..7")

    scale_t = torch.as_tensor(scale, device=shadow_weight.device, dtype=shadow_weight.dtype)
    zero_t = torch.as_tensor(zero, device=shadow_weight.device, dtype=shadow_weight.dtype)
    if scale_t.numel() != 1 or float(scale_t.item()) == 0:
        raise ValueError("weight scale must be a nonzero scalar")
    target = torch.clamp(torch.round(shadow_weight * scale_t + zero_t), qmin, qmax).to(torch.int64)
    target_flat = target.reshape(-1)
    mag_flat = mag.reshape(-1)
    valid, safe, group_index = _mapping_index(members, target.numel(), target.shape)
    grouped_target = target_flat[safe]
    grouped_mag = mag_flat[safe]

    if fixed_group_sign is None or fixed_group_shift is None:
        best_error = None
        best_state = torch.zeros(members.shape[0], dtype=torch.int64, device=members.device)
        for state_id, (sign, shift) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            candidate = fsdb_decode(
                grouped_mag,
                torch.full_like(grouped_mag, sign),
                torch.full_like(grouped_mag, shift),
            )
            error = ((candidate - grouped_target).to(torch.float64).square() * valid).sum(-1)
            if best_error is None:
                best_error = error
            else:
                better = error < best_error
                best_error = torch.where(better, error, best_error)
                best_state = torch.where(better, state_id, best_state)
        group_sign = (best_state // 2).to(torch.uint8)
        group_shift = (best_state % 2).to(torch.uint8)
    else:
        group_sign = torch.as_tensor(fixed_group_sign, device=members.device, dtype=torch.uint8).reshape(-1)
        group_shift = torch.as_tensor(fixed_group_shift, device=members.device, dtype=torch.uint8).reshape(-1)
        if group_sign.numel() != members.shape[0] or group_shift.numel() != members.shape[0]:
            raise ValueError("fixed group control count mismatch")
        if not bool(((group_sign <= 1) & (group_shift <= 1)).all()):
            raise ValueError("fixed sign/shift must be binary")

    grouped_q = fsdb_decode(
        grouped_mag,
        group_sign[:, None].expand_as(grouped_mag),
        group_shift[:, None].expand_as(grouped_mag),
    )
    q_flat = torch.empty_like(target_flat)
    q_flat[safe[valid]] = grouped_q[valid]
    q = q_flat.reshape(target.shape)
    decoded = fsdb_decode(mag, group_sign[group_index], group_shift[group_index])
    if not torch.equal(decoded, q):
        raise RuntimeError("mapping-aware FSDB encode/decode round trip failed")
    # Match the corrected forward-only reference exactly: dequantize with the
    # persistent FP64 quantization scalars, then cast once to the shadow dtype.
    scale_dequant = torch.as_tensor(scale, device=shadow_weight.device, dtype=torch.float64)
    zero_dequant = torch.as_tensor(zero, device=shadow_weight.device, dtype=torch.float64)
    hard_weight = ((q.to(torch.float64) - zero_dequant) / scale_dequant).to(shadow_weight.dtype)
    return QATHardProjection(
        q=q,
        target=target,
        hard_weight=hard_weight,
        mag=mag,
        group_sign=group_sign,
        group_shift=group_shift,
        group_index=group_index,
        ratio=ratio,
    )


class MappingAwareFrozenMagQATConv2dWrapper(FrozenMagQATConv2dWrapper):
    """Frozen-mag QAT convolution with persistent logical group membership."""

    def __init__(
        self,
        module: nn.Conv2d,
        layer_name: str,
        ratio: int,
        group_members: torch.Tensor,
        mapping_id: str,
        ibit: int = 8,
        wbit: int = 8,
    ) -> None:
        super().__init__(module, layer_name, ratio, ibit=ibit, wbit=wbit)
        members = _validated_members(group_members, module.weight.numel(), int(ratio))
        self.register_buffer("group_members", members, persistent=True)
        self.mapping_id = str(mapping_id)

    def hard_projection(self) -> QATHardProjection:
        if "frozen_mag" not in self._buffers:
            raise RuntimeError(f"ROM mag not frozen for {self.layer_name}")
        kwargs = {}
        if self.freeze_group_control:
            if "frozen_group_sign" not in self._buffers:
                raise RuntimeError(f"frozen group control not initialized for {self.layer_name}")
            kwargs = {
                "fixed_group_sign": self.frozen_group_sign,
                "fixed_group_shift": self.frozen_group_shift,
            }
        projection = project_mapping_fixed_mag(
            self.shadow_weight.detach(),
            self.frozen_mag,
            self.weight_scale,
            self.weight_zero,
            self.group_members,
            self.ratio,
            -(2 ** (self.wbit - 1)),
            2 ** (self.wbit - 1) - 1,
            **kwargs,
        )
        self._last_projection = projection
        return projection

    def export_fields(self) -> Dict[str, object]:
        fields = super().export_fields()
        fields["mapping_id"] = self.mapping_id
        fields["group_members"] = self.group_members.detach().cpu()
        return fields


class MappingAwareFrozenMagQATModelWrapper(FrozenMagQATModelWrapper):
    """R18 legacy-16 model wrapper for fixed, externally constructed mappings."""

    STATE_VERSION = 3

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, object],
        group_members_by_layer: Dict[str, torch.Tensor],
        mapping_id: str,
    ) -> None:
        config_with_groups = dict(config)
        config_with_groups["_group_members_by_layer"] = group_members_by_layer
        config_with_groups["mapping_id"] = str(mapping_id)
        super().__init__(model, config_with_groups)
        self.config.pop("_group_members_by_layer", None)

    def _replace_modules(self, selected: tuple[str, ...]) -> None:
        if tuple(selected) != R18_LEGACY16_LAYERS:
            raise ValueError("mapping-aware CIFAR QAT requires the exact R18 legacy-16 scope")
        groups = self.config.get("_group_members_by_layer")
        if not isinstance(groups, dict) or set(groups) != set(selected):
            raise ValueError("group member layer set does not match selected layers")
        mapping_id = str(self.config.get("mapping_id", ""))
        replacements = []
        for name in selected:
            module = self.model.get_submodule(name)
            if not isinstance(module, nn.Conv2d):
                raise TypeError(f"selected layer is not Conv2d: {name}")
            replacements.append((name, MappingAwareFrozenMagQATConv2dWrapper(
                module,
                name,
                self.ratio,
                groups[name],
                mapping_id,
                ibit=int(self.config.get("ibit", 8)),
                wbit=int(self.config.get("wbit", 8)),
            )))
        for name, wrapper in replacements:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(self.model.get_submodule(parent_name), child_name, wrapper)
        self._selected_layer_names = list(selected)

