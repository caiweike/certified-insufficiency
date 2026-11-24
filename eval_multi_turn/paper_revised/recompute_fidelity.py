"""Summarize TypeA FREE/REQ outcomes by unanimous Cert-100 fidelity.

Usage:
  python recompute_fidelity.py                          # defaults to repo root (../..)
  python recompute_fidelity.py cscwd_submission_bundle.zip
  python recompute_fidelity.py /path/to/certified-insufficiency
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from recompute_rq3 import BundleSource, csv_rows, load_tracks, member


def load_faithful_ids(bundle):
    key = json.loads(member(bundle, "_unified_audit_key_DO_NOT_OPEN.json"))
    items = key["cert_100"]["items"]
    votes = []
    for rater in ("cai", "tang", "yu"):
        rows = csv_rows(member(bundle, f"cert_100_{rater}.csv").decode("utf-8-sig"))
        labels = {row["audit_id"]: row["human_verdict"] for row in rows}
        assert len(rows) == len(labels) == 100 and set(labels) == set(items)
        votes.append(labels)
    return {
        item["item_id"]: item["mech_short"] for audit_id, item in items.items()
        if all(labels[audit_id] == "Faithful" for labels in votes)
    }


def summarize(tracks, selected_items, expected_cell_size):
    cells = defaultdict(Counter)
    for track in tracks:
        if track["verified_mechanism"] != "TypeA_UniqueValue":
            continue
        if track["protocol"] not in ("DYN_FREE", "DYN_REQUEST"):
            continue
        if selected_items is not None and track["item_id"] not in selected_items:
            continue
        counts = cells[(track["model_id"], track["protocol"])]
        counts["n"] += 1
        for side in ("L", "U"):
            commit = track[f"e_{side}"]
            correct = track["final_grading"][f"y_{side}"]
            assert commit in (0, 1) and correct in (0, 1)
            counts[f"E_{side}"] += commit
            counts[f"gap_{side}"] += commit * correct

    assert len(cells) == 6
    for counts in cells.values():
        assert counts["n"] == expected_cell_size
    return {
        "range_across_cell_endpoints_percent": {
            metric: [
                round(100 * min(counts[f"{metric}_L"] / counts["n"] for counts in cells.values()), 1),
                round(100 * max(counts[f"{metric}_U"] / counts["n"] for counts in cells.values()), 1),
            ]
            for metric in ("E", "gap")
        },
        "cells": {
            f"{model}:{protocol}": dict(counts)
            for (model, protocol), counts in sorted(cells.items())
        },
    }


def main():
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_bundle", nargs="?", default=default_root, type=Path)
    args = parser.parse_args()
    with BundleSource(args.original_bundle) as bundle:
        selected_items = load_faithful_ids(bundle)
        tracks = load_tracks(bundle).values()
    selected_counts = Counter(selected_items.values())
    assert selected_counts == {"TypeA": 53, "TypeB": 22}
    result = {
        "unanimous_faithful_test_items_by_regime": dict(selected_counts),
        "full_TypeA": summarize(tracks, None, 204),
        "unanimous_faithful_TypeA": summarize(tracks, selected_items, 159),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
