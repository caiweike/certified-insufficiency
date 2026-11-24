"""Recompute the human Repair-100 final-turn audit from a bundle ZIP or repo directory.

Usage:
  python recompute_rq3.py                          # defaults to repo root (../..)
  python recompute_rq3.py cscwd_submission_bundle.zip
  python recompute_rq3.py /path/to/certified-insufficiency
"""

import argparse
import csv
import gzip
import io
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


class BundleSource:
    """Unified read-only interface over either a ZIP archive or a directory tree."""

    def __init__(self, path: Path):
        self.path = path
        self.is_zip = path.is_file() and path.suffix.lower() == ".zip"
        self._zf = ZipFile(path) if self.is_zip else None
        if not self.is_zip:
            self._files = [
                (p.relative_to(path).as_posix(), p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and ".git/" not in p.relative_to(path).as_posix()
            ]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._zf is not None:
            self._zf.close()

    def namelist(self) -> list[str]:
        if self.is_zip:
            return self._zf.namelist()
        return [rel for rel, _ in self._files]

    def read(self, rel_path: str) -> bytes:
        if self.is_zip:
            return self._zf.read(rel_path)
        return (self.path / rel_path).read_bytes()


def member(bundle, name: str) -> bytes:
    matches = [
        path for path in bundle.namelist()
        if (path == name or path.endswith("/" + name))
        and not path.startswith("release/")
    ]
    if len(matches) != 1:
        # Fallback without release/ filter if caller pointed directly at a zip with prefixed root
        matches = [path for path in bundle.namelist() if path == name or path.endswith("/" + name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one bundle member for {name}, found {len(matches)}: {matches}")
    return bundle.read(matches[0])


def csv_rows(contents: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(contents)))


def load_tracks(bundle) -> dict:
    tracks = {}
    names = bundle.namelist()
    gz_matches = [
        n for n in names
        if re.search(r"(^|/)judged_test100_.*_v13_blind\.jsonl\.gz$", n)
        and not n.startswith("release/")
    ]
    raw_matches = [
        n for n in names
        if re.search(r"(^|/)judged_test100_.*_v13_blind\.jsonl$", n)
        and not n.startswith("release/")
    ]
    selected = gz_matches if len(gz_matches) == 6 else raw_matches
    if len(selected) != 6:
        selected = [n for n in names if re.search(r"(^|/)judged_test100_.*_v13_blind\.jsonl(\.gz)?$", n)]
    for name in sorted(selected):
        raw = bundle.read(name)
        if name.endswith(".gz"):
            raw = gzip.decompress(raw)
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["model_id"], row["trajectory_id"]
            if key in tracks:
                raise ValueError(f"Duplicate trajectory: {key}")
            tracks[key] = row
    assert len(tracks) == 5400, f"Expected 5400 tracks, got {len(tracks)}"
    return tracks


def anchor(track):
    earlier = next(
        turn for turn in track["turns"]
        if not turn["is_final_turn"] and turn.get("e_U") == 1
    )
    final, = (turn for turn in track["turns"] if turn["is_final_turn"])
    return earlier, final


def audit_retractions(bundle):
    tracks = load_tracks(bundle)
    eligible = {key: row for key, row in tracks.items() if row["e_U"] == 1}
    assert len(eligible) == 489
    key = json.loads(member(bundle, "_unified_audit_key_DO_NOT_OPEN.json"))["repair_100"]
    html = member(bundle, "unified_audit_tool_seed2026.html").decode("utf-8")
    payload = json.loads(re.search(r"^const D = (\{.*\});$", html, re.M).group(1))
    pairs = {pair["id"]: pair for pair in payload["repair"]}
    human = {
        name: {row["audit_id"]: row for row in csv_rows(member(
            bundle, f"repair_100_{name}.csv"
        ).decode("utf-8-sig"))}
        for name in ("cai", "tang", "yu")
    }
    items = {
        row["item_id"]: row for row in (
            json.loads(line) for line in member(bundle, "frozen_test_100.jsonl").splitlines() if line.strip()
        )
    }
    assert len(pairs) == 100
    assert set(pairs) == set(key["trajectories"])
    assert all(set(labels) == set(pairs) for labels in human.values())

    legacy_terminal = Counter()
    flagged = set()
    nonterminal_only = set()
    for track_id, track in eligible.items():
        _, final = anchor(track)
        repair = track["repair"]
        events = [
            repair[f"{judge}_llm"]["retraction_turn"]
            for judge in ("primary", "secondary") if repair[f"{judge}_retracted"]
        ]
        if events:
            flagged.add(track_id)
            legacy_terminal["flagged_anywhere"] += 1
            legacy_terminal["terminal_either"] += final["turn_idx"] in events
            legacy_terminal["terminal_both"] += events == [final["turn_idx"]] * 2
            if all(t < final["turn_idx"] for t in events):
                nonterminal_only.add(track_id)
    legacy_terminal["nonterminal_only"] = len(nonterminal_only)
    empty_keys = {
        (row["model_id"], row["trajectory_id"])
        for row in key["dual_neg_empty_trajectories"]
    }
    assert len(empty_keys) == 6 and not empty_keys & flagged
    assert all(not anchor(eligible[k])[1]["visible_reply"].strip() for k in empty_keys)

    groups = {}
    human_positive_cases = []
    sampled_flagged = set()
    for audit_id, row in key["trajectories"].items():
        track_id = row["model_id"], row["trajectory_id"]
        track = eligible[track_id]
        earlier, final = anchor(track)
        pair = pairs[audit_id]
        assert pair["early_turn"] == row["early_turn"] == earlier["turn_idx"]
        assert pair["early_reply"] == earlier["visible_reply"].strip()
        assert pair["final_reply"] == (final["visible_reply"].strip() or "(空回复)")
        item = items[track["item_id"]]
        shards = {shard["shard_id"]: shard["shard"] for shard in item["shards"]}
        assert pair["clauses"] == "\n".join(
            f"[第{index}条/{sid}] {shards[sid]}"
            for index, sid in enumerate(track["disclosure_order"], 1)
        )
        votes = []
        for labels in human.values():
            vote = labels[audit_id]
            assert int(vote["early_turn"]) == pair["early_turn"]
            assert vote["human_retracted"] in {"Yes", "No"}
            votes.append(vote["human_retracted"] == "Yes")
        is_flagged = track_id in flagged
        assert is_flagged == (row["judge_group"] != "dual_neg")
        if is_flagged:
            sampled_flagged.add(track_id)
        group = "legacy_screen_positive" if is_flagged else "sampled_screen_negative"
        regime = row["mech_short"]
        names = ["all_audited", group, regime]
        if track_id in nonterminal_only:
            names.append("legacy_nonterminal_only")
        for name in names:
            result = groups.setdefault(name, Counter())
            result["n"] += 1
            result["human_unanimous_yes"] += all(votes)
            result["human_any_yes"] += any(votes)
            result["human_majority_yes"] += sum(votes) >= 2
        if any(votes):
            human_positive_cases.append({
                "audit_id": audit_id,
                "regime": regime,
                "sampling_stratum": group,
                "human_yes": sum(votes),
            })
    assert sampled_flagged == flagged
    assert groups["legacy_screen_positive"]["n"] == 26
    assert groups["sampled_screen_negative"]["n"] == 74
    return {
        "eligible_trajectories": len(eligible),
        "empty_final_replies_outside_audit": len(empty_keys),
        "historical_full_dialogue_screen": dict(legacy_terminal),
        "human_audit": {name: dict(result) for name, result in groups.items()},
        "human_positive_cases": human_positive_cases,
    }


def main():
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_bundle", nargs="?", default=default_root, type=Path)
    args = parser.parse_args()
    with BundleSource(args.original_bundle) as bundle:
        result = audit_retractions(bundle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
