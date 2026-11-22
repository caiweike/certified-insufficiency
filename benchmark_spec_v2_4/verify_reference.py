"""Independent acceptance of the delivered reference core, not production publishing.

The production selector MUST NOT invoke this subprocess-based test runner.
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import reference_rules_v2_3 as r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comp-parquet", required=True)
    p.add_argument("--cee-parquet", required=True)
    p.add_argument("--audit-json", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    checks = []
    def check(ok, name):
        if not ok:
            raise RuntimeError(name)
        checks.append(name)
    sources = r.load_sources(args.comp_parquet, args.cee_parquet)
    audit = r.load_audit(args.audit_json, sources)
    source_map = {x["item_id"]: x for x in sources}
    files = r.build(sources, audit)
    check(files == r.build(list(reversed(sources)), audit), "source enumeration order invariance")
    ledger = list(csv.DictReader(io.StringIO(files["selection_eligibility_ledger.csv"].decode("utf-8"), newline="")))
    check(len(ledger) == 1648 and list(ledger[0]) == r.FIELDS, "ledger rows and exact header")
    check([x["item_id"] for x in ledger] == [x["item_id"] for x in sources], "ledger source bijection and numeric order")
    check(len(files["selection_eligibility_ledger.csv"].splitlines()) == 1649, "ledger 1649 physical lines")
    ledger_map = {x["item_id"]: x for x in ledger}
    selected = {}
    for status, name in (("SELECTED_TEST", "provisional_test_100.jsonl"), ("SELECTED_DEV", "provisional_dev_20.jsonl"), ("ELIGIBLE_RESERVE", "eligible_reserve_pool.jsonl")):
        values = [r.strict_json(x) for x in files[name].splitlines()]
        selected[status] = values
        check(len(values) == len({x["item_id"] for x in values}) and {x["item_id"] for x in values} == {x["item_id"] for x in ledger if x["status_code"] == status}, "membership bijection " + status)
        for item in values:
            original = source_map[item["item_id"]]
            check(all(r.csv_value(item[k]) == ledger_map[item["item_id"]][k] for k in r.FIELDS), "shared fields " + item["item_id"])
            check(item["question"] == original["question"] and item["final_answer"] == original["final_answer"][0] and item["solution"] == original["solution"] and item["unit"] == original["unit"] and item["error"] == original["error"] and item["answer_type"] == original["answer_type"], "source fidelity " + item["item_id"])
            check(r.sha(item["question"].encode("utf-8")) == item["question_sha256"], "question hash " + item["item_id"])
            check(item["effective_is_multiple_answer"] is False, "effective flag " + item["item_id"])
    combined = sum(selected.values(), [])
    check(len({x["item_id"] for x in combined}) == len(combined) and len({x["stem_cluster_id"] for x in combined}) == len(combined) and all(x["item_id"] == x["cluster_representative_id"] for x in combined), "split and cluster uniqueness")
    for name in r.BUCKETS:
        mechanism = name.split("__")[1]
        for status, expected in zip(("SELECTED_TEST", "SELECTED_DEV"), r.QUOTAS[mechanism]):
            values = [x for x in selected[status] if r.bucket(x) == name]
            check(len(values) == expected and [x["slot_index"] for x in values] == list(range(1, expected + 1)), "quota and slot order " + name + status)
        values = [x for x in selected["ELIGIBLE_RESERVE"] if r.bucket(x) == name]
        check([x["reserve_rank"] for x in values] == list(range(1, len(values) + 1)), "reserve rank and order " + name)
    for item in ledger:
        excluded = item["status_code"] == "EXCLUDED"
        check(bool(item["exclusion_stage"]) == excluded and bool(item["exclusion_reason"]) == excluded, "exclusion field closure " + item["item_id"])
        if item["exclusion_stage"] in ("STAGE_1", "STAGE_2", "STAGE_3"):
            check(item["stem_cluster_id"] == item["cluster_representative_id"] == "", "early cluster emptiness " + item["item_id"])
        check(all(item[k] != "" for k in ("macro_subfield", "provisional_mechanism", "estimated_clause_count")), "precomputed labels " + item["item_id"])
        check(bool(item["reserve_rank"]) == (item["status_code"] == "ELIGIBLE_RESERVE"), "reserve nullability " + item["item_id"])
    manifest = r.strict_json(files["selection_manifest.json"])
    for name, info in manifest["outputs"].items():
        check(r.sha(files[name]) == info["sha256"] and len(files[name]) == info["bytes"], "manifest data hash " + name)
    check(files["selection_manifest.json.sha256"] == (r.sha(files["selection_manifest.json"]) + "\n").encode("ascii"), "manifest sidecar hash")
    check(all(b"\r" not in value and not value.startswith(b"\xef\xbb\xbf") for value in files.values()), "LF without BOM")
    check(r.multi_subquestion("(1)求a的值；(2)求b的值"), "ASCII question markers")
    check(r.multi_subquestion("（1）求a的值；（2）求b的值"), "fullwidth question markers")
    check(not r.multi_subquestion(source_map["ob-comp-289"]["question"]), "function arguments are not question markers")
    for id_ in ("ob-comp-700", "ob-comp-530"):
        question = source_map[id_]["question"]
        check(not r.multi_subquestion(question) and r.split_stem_and_target(question)[1].startswith("求最小的") and r.classify_provisional_mechanism(question) == r.TYPE_B, "leading target " + id_)
    check(r.classify_provisional_mechanism(source_map["ob-comp-256"]["question"]) == r.TYPE_B, "extremum symbol coreference")
    check(r.classify_provisional_mechanism("已知函数f(x)=x^2+a，最大值为4，求a的值。") == r.TYPE_A, "prior extremum is not automatically the target")
    pairs = [("x=1.2", "x=12"), ("A(1,23)", "A(12,3)"), ("x:y=2:3", "xy=23"), ("1 2", "12"), ("x-y", "x+y"), (r"\mathbf{x}", r"\mathrm{x}")]
    for left, right in pairs:
        check(r.math_tokens(left) != r.math_tokens(right), "math token distinction " + left)
    check(r.math_tokens("$x = 1.2$") == r.math_tokens("x=1.2"), "format-only token equivalence")
    check(r.scalar_answer_has_multiple_values("$(1,2)$") and r.scalar_answer_has_multiple_values(r"$1\pm2$") and not r.scalar_answer_has_multiple_values("$f(m,n)=m+n$") and not r.scalar_answer_has_multiple_values(r"$\frac{1}{50}；$"), "scalar answer grouping")
    check("ob-comp-230" not in audit["exclusions"], "remove unsupported overdetermination exclusion")
    check(all(ledger_map[k]["status_code"] == "EXCLUDED" for k in audit["exclusions"]), "audit exclusions applied")
    check(any(x["item_id"] == "ob-cee-3106" and x["source_flag"] is True and x["effective_flag"] is False for x in manifest["audit"]["flag_corrections_applied"]), "flag correction provenance")
    duplicate_rejected = False
    try:
        r.strict_json(b'{"x":1,"x":2}')
    except r.SpecError:
        duplicate_rejected = True
    check(duplicate_rejected, "duplicate JSON keys rejected")
    # Separate driver: it never invokes this verifier, preventing recursion.
    child_results = []
    with tempfile.TemporaryDirectory(prefix="ob_spec_v23_") as temp:
        for seed, optimize in (("0", False), ("42", False), ("42", True)):
            dest = Path(temp) / (seed + ("_optimized" if optimize else "_normal"))
            cmd = [sys.executable, "-B"] + (["-O"] if optimize else []) + [str(Path(__file__).with_name("recompute_baseline.py")),
                   "--comp-parquet", str(Path(args.comp_parquet).resolve()), "--cee-parquet", str(Path(args.cee_parquet).resolve()),
                   "--audit-json", str(Path(args.audit_json).resolve()), "--output", str(dest)]
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            process = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
            check(process.returncode == 0, "subprocess exit " + str((seed, optimize)) + process.stderr)
            check(all((dest / name).read_bytes() == content for name, content in files.items()), "six-file byte identity " + str((seed, optimize)))
            child_results.append({"pythonhashseed": seed, "optimized": optimize, "all_six_files_equal": True})
        retained_path = Path(temp) / "retained_fixture.jsonl"
        retained_path.write_bytes(b"".join(r.canonical_json({"id": x["item_id"], "question_sha256": x["question_sha256"]}) for x in selected["SELECTED_TEST"]))
        ids, meta = r.load_retained(retained_path, sources)
        retained_files = r.build(sources, audit, ids, meta)
        retained_test = [r.strict_json(line) for line in retained_files["provisional_test_100.jsonl"].splitlines()]
        check({x["item_id"] for x in retained_test} == ids, "retention fixture preserves all eligible in-quota test ids")
    report = {"scope": "reference pure core; production CLI publication and stage2 remain developer acceptance obligations", "result": "PASS", "check_count": len(checks), "manifest_sha256": r.sha(files["selection_manifest.json"]), "subprocess_runs": child_results,
              "platforms_actually_tested": [{"platform": sys.platform, "python": sys.version.split()[0]}],
              "not_executed": ["production CURRENT publication and crash recovery", "other operating systems or Python versions", "stage2 certificate verification"]}
    Path(args.report).write_bytes(r.canonical_json(report))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
