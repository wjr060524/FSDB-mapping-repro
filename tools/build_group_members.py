"""Small entry point for the extracted historical mapping implementation.

Input is a torch-saved dictionary with a ``layers`` dictionary and a
``selected_layers`` sequence.  Each layer must contain the exact arrays
documented in README.md.  This tool does not train or evaluate a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fsdb_mapping_repro.mapping_methods import CANDIDATES, build_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer-state", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ratio = 4 if args.method == "e_state_oc" else 16
    # The historical layer dictionaries contain NumPy arrays.  Load only a
    # trusted, locally generated artifact because this is pickle-backed.
    state = torch.load(args.layer_state, map_location="cpu", weights_only=False)
    layers = state["layers"]
    selected = tuple(state["selected_layers"])
    groups, record = build_groups(layers, selected, args.method, ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"group_members": groups, "record": record}, args.output)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
