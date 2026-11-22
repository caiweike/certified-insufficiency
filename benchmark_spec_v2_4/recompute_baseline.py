"""Read-only source audit and golden-byte generator, separate from production 01.

Writes exclusively to the explicitly supplied new --output directory. It does
not implement locking, CURRENT publication or stage-two certificate checking.
"""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import reference_rules_v2_3 as rules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-parquet", required=True)
    parser.add_argument("--cee-parquet", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefer-retained")
    parser.add_argument("--test-seed", type=int, default=2026)
    parser.add_argument("--dev-seed", type=int, default=2027)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("baseline destination must not already exist")
    sources = rules.load_sources(args.comp_parquet, args.cee_parquet)
    audit = rules.load_audit(args.audit_json, sources)
    retained_ids, retained_meta = rules.load_retained(args.prefer_retained, sources)
    files = rules.build(sources, audit, retained_ids, retained_meta, args.test_seed, args.dev_seed)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (destination / name).write_bytes(content)
    manifest = rules.strict_json(files["selection_manifest.json"])
    report = {
        "report_type": "reference_core_baseline_not_production_script_acceptance",
        "environment": {"python": platform.python_version(), "os": platform.system(), "pyarrow": importlib.metadata.version("pyarrow")},
        "parameters": manifest["parameters"], "rules_sha256": manifest["rules_sha256"],
        "audit_sha256": rules.AUDIT_SHA256,
        "manifest_sha256": rules.sha(files["selection_manifest.json"]),
        "file_sha256": {name: rules.sha(content) for name, content in sorted(files.items())},
        "status_counts": manifest["status_counts"], "buckets": manifest["buckets"], "funnel": manifest["funnel"],
    }
    (destination / "baseline_report.json").write_bytes(rules.canonical_json(report))
    print(json.dumps({"manifest_sha256": report["manifest_sha256"], "status_counts": report["status_counts"], "buckets": report["buckets"], "funnel": report["funnel"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
