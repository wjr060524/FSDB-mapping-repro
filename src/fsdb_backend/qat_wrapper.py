"""Trainable FSDB wrappers with floating shadow weights and hard forwards."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F

from .qat_quantizer import (
    QATHardProjection,
    project_qat_weight,
    project_qat_weight_fixed_group_control,
)


R18_LEGACY16_LAYERS = tuple(
    f"layer{stage}.{block}.conv{conv}"
    for stage, blocks in ((1, 2), (2, 2), (3, 2), (4, 2))
    for block in range(blocks)
    for conv in (1, 2)
)


class FSDBQATConv2dWrapper(nn.Module):
    def __init__(
        self, module: nn.Conv2d, layer_name: str, ratio: int,
        ibit: int = 8, wbit: int = 8, freeze_group_control: bool = False,
    ):
        super().__init__()
        self.module = module
        self.layer_name = layer_name
        self.ratio = int(ratio)
        self.ibit = int(ibit)
        self.wbit = int(wbit)
        self.runtime_mode = "qat"
        self.freeze_group_control = bool(freeze_group_control)
        self.register_buffer("weight_scale", torch.empty(0, dtype=torch.float64))
        self.register_buffer("weight_zero", torch.empty(0, dtype=torch.float64))
        self.register_buffer("input_scale", torch.empty(0, dtype=torch.float64))
        self.register_buffer("input_zero", torch.empty(0, dtype=torch.float64))
        self.register_buffer("input_qmin", torch.empty(0, dtype=torch.int64))
        self.register_buffer("input_qmax", torch.empty(0, dtype=torch.int64))
        self._last_projection: QATHardProjection | None = None

    def lock_group_control(self, group_sign: torch.Tensor, group_shift: torch.Tensor) -> None:
        if self.ratio == 1:
            raise ValueError("group-control locking is only meaningful for grouped FSDB")
        sign = torch.as_tensor(group_sign, dtype=torch.uint8, device=self.module.weight.device).detach().clone()
        shift = torch.as_tensor(group_shift, dtype=torch.uint8, device=self.module.weight.device).detach().clone()
        if "frozen_group_sign" in self._buffers:
            self.frozen_group_sign = sign
            self.frozen_group_shift = shift
        else:
            self.register_buffer("frozen_group_sign", sign, persistent=True)
            self.register_buffer("frozen_group_shift", shift, persistent=True)
        self.freeze_group_control = True

    @property
    def shadow_weight(self) -> nn.Parameter:
        return self.module.weight

    def import_quant_params(self, params: Dict[str, object]) -> None:
        required = {"weight_scale", "weight_zero", "input_scale", "input_zero", "input_qmin", "input_qmax"}
        missing = sorted(required - set(params))
        if missing:
            raise KeyError(f"missing quant parameters for {self.layer_name}: {missing}")
        device = self.module.weight.device
        for name in ("weight_scale", "weight_zero", "input_scale", "input_zero"):
            setattr(self, name, torch.as_tensor(params[name], dtype=torch.float64, device=device).detach().clone())
        for name in ("input_qmin", "input_qmax"):
            setattr(self, name, torch.as_tensor(params[name], dtype=torch.int64, device=device).detach().clone())

    def _activation_ste(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale.numel() == 0:
            raise RuntimeError(f"quant parameters not imported for {self.layer_name}")
        scale = self.input_scale.to(dtype=x.dtype)
        zero = self.input_zero.to(dtype=x.dtype)
        qmin = int(self.input_qmin.item()) if float(x.detach().min().item()) < 0 else 0
        q = torch.clamp(torch.round(x * scale + zero), qmin, int(self.input_qmax.item()))
        hard = (q - zero) / scale
        # Numerically exactly hard while retaining identity gradient to x.
        return hard.detach() + (x - x.detach())

    def hard_projection(self) -> QATHardProjection:
        if self.weight_scale.numel() == 0:
            raise RuntimeError(f"quant parameters not imported for {self.layer_name}")
        arguments = (
            self.module.weight.detach(), self.weight_scale, self.weight_zero,
            -(2 ** (self.wbit - 1)), 2 ** (self.wbit - 1) - 1, self.ratio,
        )
        if self.freeze_group_control:
            if "frozen_group_sign" not in self._buffers:
                raise RuntimeError(f"frozen group control not initialized for {self.layer_name}")
            projection = project_qat_weight_fixed_group_control(
                *arguments, self.frozen_group_sign, self.frozen_group_shift,
            )
        else:
            projection = project_qat_weight(*arguments)
        self._last_projection = projection
        return projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.runtime_mode == "float":
            return self.module(x)
        projection = self.hard_projection()
        hard = projection.hard_weight
        # Algebraically the usual shadow + (hard - shadow).detach(), but this
        # ordering avoids a possible one-ULP cancellation error in the forward.
        weight_ste = hard.detach() + (self.module.weight - self.module.weight.detach())
        x_ste = self._activation_ste(x)
        return F.conv2d(
            x_ste, weight_ste, self.module.bias, self.module.stride,
            self.module.padding, self.module.dilation, self.module.groups,
        )

    def export_fields(self) -> Dict[str, object]:
        projection = self.hard_projection()
        return {
            "layer_name": self.layer_name,
            "ratio": self.ratio,
            "shape": list(self.module.weight.shape),
            "mag": projection.mag.detach().cpu(),
            "group_sign": projection.group_sign.detach().cpu(),
            "group_shift": projection.group_shift.detach().cpu(),
            "group_index": projection.group_index.detach().cpu(),
            "q": projection.q.detach().cpu(),
            "weight_scale": self.weight_scale.detach().cpu(),
            "weight_zero": self.weight_zero.detach().cpu(),
        }


class FSDBQATModelWrapper(nn.Module):
    STATE_VERSION = 1

    def __init__(self, model: nn.Module, config: Dict[str, object]):
        super().__init__()
        self.model = model
        self.config = dict(config)
        self.ratio = int(config.get("rom_ram_ratio", 1))
        selected = tuple(config.get("selected_layer_names", R18_LEGACY16_LAYERS))
        self._replace_modules(selected)

    def _replace_modules(self, selected: tuple[str, ...]) -> None:
        selected_set = set(selected)
        replacements = []
        for name, module in self.model.named_modules():
            if name in selected_set:
                if not isinstance(module, nn.Conv2d):
                    raise TypeError(f"selected layer is not Conv2d: {name}")
                replacements.append((name, FSDBQATConv2dWrapper(
                    module, name, self.ratio,
                    ibit=int(self.config.get("ibit", 8)),
                    wbit=int(self.config.get("wbit", 8)),
                    freeze_group_control=bool(self.config.get("freeze_group_control", False)),
                )))
        found = {name for name, _ in replacements}
        if found != selected_set:
            raise ValueError(f"selected layer mismatch: missing={sorted(selected_set - found)}")
        for name, wrapper in replacements:
            parent_name, child_name = name.rsplit(".", 1)
            parent = self.model.get_submodule(parent_name)
            setattr(parent, child_name, wrapper)
        self._selected_layer_names = list(selected)

    def fsdb_layers(self) -> Iterable[FSDBQATConv2dWrapper]:
        for module in self.modules():
            if isinstance(module, FSDBQATConv2dWrapper):
                yield module

    @property
    def selected_layer_names(self) -> List[str]:
        return list(self._selected_layer_names)

    def import_quant_params(self, params_by_layer: Dict[str, Dict[str, object]]) -> None:
        if set(params_by_layer) != set(self._selected_layer_names):
            raise ValueError("quant parameter layer set does not match selected QAT layers")
        for layer in self.fsdb_layers():
            layer.import_quant_params(params_by_layer[layer.layer_name])

    def set_runtime_mode(self, mode: str) -> None:
        if mode not in ("qat", "float"):
            raise ValueError(f"invalid runtime mode: {mode}")
        for layer in self.fsdb_layers():
            layer.runtime_mode = mode

    def lock_group_controls(self, controls_by_layer: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if set(controls_by_layer) != set(self._selected_layer_names):
            raise ValueError("group-control layer set does not match selected QAT layers")
        for layer in self.fsdb_layers():
            controls = controls_by_layer[layer.layer_name]
            layer.lock_group_control(controls["sign"], controls["shift"])
        self.config["freeze_group_control"] = True

    def export_quant_state(self) -> Dict[str, object]:
        return {
            "format": "fsdb_qat",
            "version": self.STATE_VERSION,
            "ratio": self.ratio,
            "config": deepcopy(self.config),
            "selected_layers": self.selected_layer_names,
            "layers": {layer.layer_name: layer.export_fields() for layer in self.fsdb_layers()},
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
