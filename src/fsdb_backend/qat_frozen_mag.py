"""Isolated CIFAR transfer QAT path with immutable ROM magnitudes.

The trainable variables are floating-point shadow weights.  At every forward,
the best shared sign/shift state is selected for each group while the ROM mag
buffer is unchanged.  The convolution sees the exact hard decoded weight;
the shadow receives the identity STE gradient.  Existing QAT paths are not
modified by importing this module.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .qat_quantizer import QATHardProjection, project_qat_weight
from .qat_wrapper import FSDBQATConv2dWrapper, FSDBQATModelWrapper, R18_LEGACY16_LAYERS
from .quantizer import fsdb_decode


def project_fixed_mag(
    shadow_weight: torch.Tensor,
    frozen_mag: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    ratio: int,
    qmin: int = -128,
    qmax: int = 127,
) -> QATHardProjection:
    """Choose binary SRAM controls with fixed 3-bit ROM mag.

    Groups are contiguous *within* each output channel.  Padded tail elements
    do not participate in state selection.  Tie order is 00, 01, 10, 11.
    """
    ratio = int(ratio)
    if ratio not in (1, 4, 16):
        raise ValueError("ratio must be 1, 4, or 16")
    if shadow_weight.shape != frozen_mag.shape or shadow_weight.ndim < 1:
        raise ValueError("frozen mag shape must equal convolution weight shape")
    mag = frozen_mag.to(device=shadow_weight.device, dtype=torch.uint8)
    if bool((mag > 7).any()):
        raise ValueError("frozen mag must be in 0..7")

    out_channels = shadow_weight.shape[0]
    per_channel = shadow_weight.numel() // out_channels
    groups_per_channel = (per_channel + ratio - 1) // ratio
    padded = groups_per_channel * ratio
    scale_t = torch.as_tensor(scale, device=shadow_weight.device, dtype=shadow_weight.dtype)
    zero_t = torch.as_tensor(zero, device=shadow_weight.device, dtype=shadow_weight.dtype)
    if scale_t.numel() != 1 or float(scale_t.item()) == 0:
        raise ValueError("weight scale must be a nonzero scalar")
    target = torch.clamp(torch.round(shadow_weight * scale_t + zero_t), qmin, qmax).to(torch.int64)
    target_flat = target.reshape(out_channels, per_channel)
    mag_flat = mag.reshape(out_channels, per_channel)
    valid = torch.arange(padded, device=mag.device).reshape(1, groups_per_channel, ratio) < per_channel
    target_padded = torch.zeros((out_channels, padded), dtype=torch.int64, device=mag.device)
    mag_padded = torch.zeros((out_channels, padded), dtype=torch.uint8, device=mag.device)
    target_padded[:, :per_channel] = target_flat
    mag_padded[:, :per_channel] = mag_flat
    grouped_target = target_padded.reshape(out_channels, groups_per_channel, ratio)
    grouped_mag = mag_padded.reshape(out_channels, groups_per_channel, ratio)

    best_error = None
    best_state = torch.zeros((out_channels, groups_per_channel), dtype=torch.int64, device=mag.device)
    states = ((0, 0), (0, 1), (1, 0), (1, 1))
    for state_id, (sign, shift) in enumerate(states):
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

    group_sign = (best_state // 2).to(torch.uint8).reshape(-1)
    group_shift = (best_state % 2).to(torch.uint8).reshape(-1)
    group_ids = torch.arange(out_channels * groups_per_channel, dtype=torch.int64, device=mag.device)
    group_index = group_ids.reshape(out_channels, groups_per_channel, 1).expand(
        -1, -1, ratio
    ).reshape(out_channels, padded)[:, :per_channel].reshape(shadow_weight.shape)
    q = fsdb_decode(mag, group_sign[group_index], group_shift[group_index])
    hard_weight = (q.to(shadow_weight.dtype) - zero_t) / scale_t
    return QATHardProjection(
        q=q, target=target, hard_weight=hard_weight, mag=mag,
        group_sign=group_sign, group_shift=group_shift,
        group_index=group_index, ratio=ratio,
    )


class FrozenMagQATConv2dWrapper(FSDBQATConv2dWrapper):
    """Existing activation QAT + STE forward, with fixed-mag projection."""

    def freeze_magnitude(self) -> None:
        if self.weight_scale.numel() != 1:
            raise RuntimeError("import quant parameters before freezing mag")
        initial = project_qat_weight(
            self.shadow_weight.detach(), self.weight_scale, self.weight_zero,
            -(2 ** (self.wbit - 1)), 2 ** (self.wbit - 1) - 1, self.ratio,
        )
        fixed = initial.mag.detach().clone().to(torch.uint8)
        if "frozen_mag" in self._buffers:
            self.frozen_mag = fixed
        else:
            self.register_buffer("frozen_mag", fixed, persistent=True)

    def hard_projection(self) -> QATHardProjection:
        if "frozen_mag" not in self._buffers:
            raise RuntimeError(f"ROM mag not frozen for {self.layer_name}")
        projection = project_fixed_mag(
            self.shadow_weight.detach(), self.frozen_mag,
            self.weight_scale, self.weight_zero, self.ratio,
            -(2 ** (self.wbit - 1)), 2 ** (self.wbit - 1) - 1,
        )
        self._last_projection = projection
        return projection


class FrozenMagQATModelWrapper(FSDBQATModelWrapper):
    """Opt-in R18 legacy-16 wrapper; default FSDB QAT remains untouched."""

    STATE_VERSION = 2

    def _replace_modules(self, selected: tuple[str, ...]) -> None:
        if tuple(selected) != R18_LEGACY16_LAYERS:
            raise ValueError("CIFAR diagnostic requires the exact R18 legacy-16 scope")
        replacements = []
        for name in selected:
            module = self.model.get_submodule(name)
            if not isinstance(module, nn.Conv2d):
                raise TypeError(f"selected layer is not Conv2d: {name}")
            replacements.append((name, FrozenMagQATConv2dWrapper(
                module, name, self.ratio,
                ibit=int(self.config.get("ibit", 8)),
                wbit=int(self.config.get("wbit", 8)),
            )))
        for name, wrapper in replacements:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(self.model.get_submodule(parent_name), child_name, wrapper)
        self._selected_layer_names = list(selected)

    def freeze_all_magnitudes(self) -> None:
        for layer in self.fsdb_layers():
            layer.freeze_magnitude()

    def mag_state(self) -> Dict[str, torch.Tensor]:
        return {
            layer.layer_name: layer.frozen_mag.detach().cpu().clone()
            for layer in self.fsdb_layers()
        }
