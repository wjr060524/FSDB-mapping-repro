"""Verbatim algorithm extraction for the two validated OC-local mappings.

The function bodies are copied from:
  run_r18_ptq_sharing_mapping_search.py:120-131, 313-424, 461-483
  run_fsdb_frozen_mag_scaling_imagenet.py:290-332

Only the registry entries and dispatch paths for ``e_state_oc`` and
``h_error_vector_oc`` are exposed.  Exact-MinRoute and all other candidates
are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import torch

from fsdb_backend.quantizer import fsdb_decode


STATES = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    kind: str
    scope: str
    feature: str = ""
    block_size: int = 0
    order: str = ""
    feasibility: str = ""
    metadata: str = ""
    alias_of: str = ""
    oracle_tier: str = "none"
    offset: int = 0


CANDIDATES = {
    "e_state_oc": Candidate(
        "e_state_oc", "E", "sort", "oc", feature="state",
        feasibility="permutation", metadata="per_layer_permutation",
        oracle_tier="output_channel",
    ),
    "h_error_vector_oc": Candidate(
        "h_error_vector_oc", "H", "sort", "oc", feature="error_vector",
        feasibility="permutation", metadata="per_layer_permutation",
    ),
}


def partitions(shape: tuple[int, ...], scope: str, block_size: int = 0) -> list[np.ndarray]:
    co, ci, kh, kw = shape
    grid = np.arange(co * ci * kh * kw, dtype=np.int64).reshape(shape)
    if scope == "global":
        return [grid.reshape(-1)]
    if scope == "oc":
        return [grid[o].reshape(-1) for o in range(co)]
    if scope == "ic":
        return [grid[:, i, :, :].reshape(-1) for i in range(ci)]
    if scope == "kernel":
        return [grid[o, i].reshape(-1) for o in range(co) for i in range(ci)]
    if scope == "oc_spatial":
        return [grid[o, :, h, w].reshape(-1) for o in range(co) for h in range(kh) for w in range(kw)]
    if scope == "spatial_global":
        return [grid[:, :, h, w].reshape(-1) for h in range(kh) for w in range(kw)]
    if scope == "ocblock":
        result = []
        for o in range(co):
            flat = grid[o].reshape(-1)
            result.extend(flat[i:i + block_size] for i in range(0, flat.size, block_size))
        return result
    if scope == "icblock":
        result = []
        for i in range(ci):
            flat = grid[:, i, :, :].reshape(-1)
            result.extend(flat[j:j + block_size] for j in range(0, flat.size, block_size))
        return result
    if scope == "globalblock":
        flat = grid.reshape(-1)
        return [flat[j:j + block_size] for j in range(0, flat.size, block_size)]
    raise ValueError(scope)


def feature_keys(layer: dict, feature: str) -> tuple[np.ndarray, ...]:
    mag = layer["mag"].reshape(-1).astype(np.int64)
    sign = layer["sign"].reshape(-1).astype(np.int64)
    shift = layer["shift"].reshape(-1).astype(np.int64)
    q = layer["q1"].reshape(-1).astype(np.int64)
    target = layer["target"].reshape(-1).astype(np.int64)
    weight = layer["weight"].reshape(-1).astype(np.float64)
    state = 2 * sign + shift
    residual = target - q
    table = {
        "state": (state, mag, target),
        "sign_shift_mag": (sign, shift, mag, target),
        "shift_sign_mag": (shift, sign, mag, target),
        "mag_state": (mag, state, target),
        "state_q": (state, q, target),
        "state_abs_weight": (state, np.abs(weight), weight),
        "state_residual": (state, np.abs(residual), residual, mag),
        "q": (q,), "abs_q": (np.abs(q), q), "target": (target,),
        "weight": (weight,), "abs_weight": (np.abs(weight), weight),
        "residual_state": (np.abs(residual), state, mag),
    }
    if feature in {"error_vector", "best_state_margin"}:
        errors = error_signature(layer)
        sorted_errors = np.sort(errors, axis=1)
        best_state = errors.argmin(axis=1)
        margin = sorted_errors[:, 1] - sorted_errors[:, 0]
        table["error_vector"] = (
            best_state, errors[:, 0], errors[:, 1], errors[:, 2], errors[:, 3], target
        )
        table["best_state_margin"] = (best_state, -margin, target)
    return table[feature]


def sorted_partitions(layer: dict, candidate: Candidate) -> list[np.ndarray]:
    result = []
    keys = feature_keys(layer, candidate.feature)
    for part in partitions(layer["shape"], candidate.scope, candidate.block_size):
        # np.lexsort uses the last key as primary.  Reverse for human order;
        # original semantic index is the final deterministic tie-breaker.
        local_keys = [part]
        local_keys.extend(key[part] for key in reversed(keys))
        order = np.lexsort(tuple(local_keys))
        ordered = part[order]
        if candidate.offset and ordered.size:
            ordered = np.roll(ordered, -min(candidate.offset, ordered.size - 1))
        result.append(ordered)
    return result


def authoritative_state_table() -> np.ndarray:
    """Build the search LUT exclusively from the authoritative backend."""
    mags = torch.arange(8, dtype=torch.int64).repeat_interleave(4)
    signs = torch.tensor([s for _ in range(8) for s, _ in STATES], dtype=torch.int64)
    shifts = torch.tensor([f for _ in range(8) for _, f in STATES], dtype=torch.int64)
    return fsdb_decode(mags, signs, shifts).reshape(8, 4).cpu().numpy()


STATE_TABLE = authoritative_state_table()


def state_values(mag: np.ndarray) -> np.ndarray:
    return STATE_TABLE[mag.astype(np.int64)]


def error_signature(layer: dict) -> np.ndarray:
    target = layer["target"].reshape(-1).astype(np.int64)
    values = state_values(layer["mag"].reshape(-1))
    return (values - target[:, None]).astype(np.float64) ** 2


def candidate_partitions(layer: dict, candidate: Candidate, ratio: int) -> list[np.ndarray]:
    # Both exported candidates use the original sort path.  Other historical
    # dispatch branches are deliberately excluded from this focused bundle.
    if candidate.kind == "sort":
        return sorted_partitions(layer, candidate)
    raise ValueError(candidate.kind)


def make_groups(parts: Iterable[np.ndarray], ratio: int) -> tuple[np.ndarray, int]:
    groups = []
    padded_count = 0
    for part in parts:
        for start in range(0, len(part), ratio):
            chunk = part[start:start + ratio]
            row = np.full(ratio, -1, dtype=np.int64)
            row[:len(chunk)] = chunk
            padded_count += ratio - len(chunk)
            groups.append(row)
    return np.stack(groups), padded_count


def build_groups(layers: dict, selected: tuple[str, ...], candidate_id: str, ratio: int):
    candidate = CANDIDATES[candidate_id]
    output = {}
    records = {}
    aggregate = hashlib.sha256(); coordinates = hashlib.sha256()
    total = group_total = padding_total = 0
    for name in selected:
        layer = layers[name]
        groups, padding = make_groups(
            candidate_partitions(layer, candidate, ratio), ratio
        )
        valid = groups >= 0
        seen = groups[valid]
        count = int(layer["weight"].size)
        if seen.size != count or np.unique(seen).size != count or seen.min() != 0 or seen.max() != count - 1:
            raise RuntimeError(f"mapping is not a bijection: {candidate_id} {name}")
        members = torch.from_numpy(groups.astype(np.int64, copy=False)).clone()
        output[name] = members
        aggregate.update(name.encode()); aggregate.update(groups.astype("<i8", copy=False).tobytes())
        group_id = np.full(count, -1, dtype=np.int64)
        slot = np.full(count, -1, dtype=np.int16)
        ids = np.broadcast_to(np.arange(groups.shape[0], dtype=np.int64)[:, None], groups.shape)
        slots = np.broadcast_to(np.arange(ratio, dtype=np.int16)[None, :], groups.shape)
        group_id[seen] = ids[valid]; slot[seen] = slots[valid]
        if bool((group_id < 0).any()) or bool((slot < 0).any()):
            raise RuntimeError("logical coordinate audit failed")
        coordinates.update(name.encode())
        coordinates.update(np.arange(count, dtype="<i8").tobytes())
        coordinates.update(group_id.astype("<i8", copy=False).tobytes())
        coordinates.update(slot.astype("<i2", copy=False).tobytes())
        records[name] = {
            "shape": list(layer["shape"]), "groups": int(groups.shape[0]),
            "padding": int(padding),
            "ordered_group_sha256": hashlib.sha256(groups.astype("<i8", copy=False).tobytes()).hexdigest(),
        }
        total += count; group_total += int(groups.shape[0]); padding_total += int(padding)
    return output, {
        "candidate_id": candidate_id, "ratio": int(ratio), "scope": candidate.scope,
        "feature": candidate.feature, "offset": candidate.offset,
        "logical_weight_count": total, "group_count": group_total,
        "padding_slots": padding_total, "ordered_mapping_sha256": aggregate.hexdigest(),
        "logical_coordinate_and_slot_sha256": coordinates.hexdigest(), "layers": records,
    }
