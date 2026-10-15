"""Import existing SELECT workload files with explicit template IDs and file hashes."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sql-dir", type=Path, required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows, hashes = [], {}
    for path in sorted(a.sql_dir.glob("*.sql")):
        text = path.read_text().strip().rstrip(";")
        if not re.match(r"\s*(SELECT|WITH)\b", text, re.I):
            raise ValueError(f"expected read-only SELECT workload: {path}")
        m = re.fullmatch(r"(\d+)[a-z]+", path.stem)
        if not m:
            raise ValueError(f"cannot infer a template ID; provide an explicit CSV: {path}")
        rows.append({"query_id": f"{path.stem}_{a.workload}", "template": f"{m.group(1)}_{a.workload}", "sql": text})
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not rows or a.out.exists():
        raise ValueError("empty input or output already exists")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "template", "sql"])
        writer.writeheader()
        writer.writerows(rows)
    a.out.with_suffix(".sources.json").write_text(json.dumps({"source_directory": str(a.sql_dir), "sha256": hashes}, indent=2) + "\n")
    print(f"{len(rows)} queries, {len({r['template'] for r in rows})} templates -> {a.out}")


if __name__ == "__main__":
    main()
