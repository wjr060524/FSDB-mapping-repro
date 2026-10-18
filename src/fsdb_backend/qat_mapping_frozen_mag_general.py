"""Architecture-generic frozen-mag FSDB wrappers for the scaling experiment.

This module is isolated from the validated R18 implementation.  It preserves
the same hard forward/identity-STE semantics while accepting an explicitly
preregistered Conv2d layer list for R34 BasicBlocks and R50 Bottlenecks.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .qat_frozen_mag import FrozenMagQATConv2dWrapper, FrozenMagQATModelWrapper
from .qat_mapping_frozen_mag import MappingAwareFrozenMagQATConv2dWrapper


def _replace_selected(model_wrapper, selected: tuple[str, ...], factory) -> None:
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected layer names must be nonempty and unique")
    replacements = []
    for name in selected:
        module = model_wrapper.model.get_submodule(name)
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"selected layer is not Conv2d: {name}")
        replacements.append((name, factory(name, module)))
    for name, wrapper in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model_wrapper.model.get_submodule(parent_name), child_name, wrapper)
    model_wrapper._selected_layer_names = list(selected)


class GeneralFrozenMagQATModelWrapper(FrozenMagQATModelWrapper):
    """Frozen-mag wrapper whose exact layer scope comes from the protocol."""

    STATE_VERSION = 4

    def _replace_modules(self, selected: tuple[str, ...]) -> None:
        _replace_selected(
            self,
            selected,
            lambda name, module: FrozenMagQATConv2dWrapper(
                module,
                name,
                self.ratio,
                ibit=int(self.config.get("ibit", 8)),
                wbit=int(self.config.get("wbit", 8)),
            ),
        )


class GeneralMappingAwareFrozenMagQATModelWrapper(FrozenMagQATModelWrapper):
    """Fixed logical-group wrapper for arbitrary preregistered Conv2d scopes."""

    STATE_VERSION = 5

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
        groups = self.config.get("_group_members_by_layer")
        if not isinstance(groups, dict) or set(groups) != set(selected):
            raise ValueError("group member layer set does not match selected layers")
        mapping_id = str(self.config.get("mapping_id", ""))
        _replace_selected(
            self,
            selected,
            lambda name, module: MappingAwareFrozenMagQATConv2dWrapper(
                module,
                name,
                self.ratio,
                groups[name],
                mapping_id,
                ibit=int(self.config.get("ibit", 8)),
                wbit=int(self.config.get("wbit", 8)),
            ),
        )
