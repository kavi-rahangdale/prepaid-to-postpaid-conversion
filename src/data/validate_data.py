"""Data validation: check the raw dataset against rules defined in configs/config.yaml.

Run from the project root:
    python -m src.data.validate_data            # report only
    python -m src.data.validate_data --strict   # exit code 1 if any ERROR (use in pipelines/CI)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.data.generate_data import ROOT, load_config

ERROR, WARN, OK = "ERROR", "WARN", "OK"


def _result(check: str, column: str, severity: str, failed: int, total: int, message: str) -> dict:
    return {
        "check": check,
        "column": column,
        "severity": severity,
        "failed_rows": int(failed),
        "failed_pct": round(100 * failed / total, 3) if total else 0.0,
        "message": message,
    }


def check_schema(df: pd.DataFrame, expected: list) -> list:
    missing = [c for c in expected if c not in df.columns]
    extra = [c for c in df.columns if c not in expected]
    results = []
    if missing:
        results.append(_result("schema", ",".join(missing), ERROR, 0, len(df), f"missing columns: {missing}"))
    if extra:
        results.append(_result("schema", ",".join(extra), WARN, 0, len(df), f"unexpected columns: {extra}"))
    if not missing and not extra:
        results.append(_result("schema", "-", OK, 0, len(df), "all expected columns present"))
    return results


def check_duplicates(df: pd.DataFrame, id_col: str) -> list:
    n = len(df)
    full = int(df.duplicated().sum())
    by_id = int(df[id_col].duplicated().sum())
    return [
        _result("duplicate_rows", "-", ERROR if full else OK, full, n, "fully identical rows"),
        _result("duplicate_ids", id_col, ERROR if by_id else OK, by_id, n, "repeated customer ids"),
    ]


def check_missing(df: pd.DataFrame, max_fraction: float) -> list:
    results = []
    n = len(df)
    for col, cnt in df.isna().sum().items():
        if cnt == 0:
            continue
        severity = WARN if cnt / n > max_fraction else OK
        results.append(_result("missing_values", col, severity, cnt, n, f"threshold {max_fraction:.0%}"))
    return results


def check_ranges(df: pd.DataFrame, ranges: dict) -> list:
    results = []
    n = len(df)
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            results.append(_result("dtype", col, ERROR, n, n, "column is not numeric"))
            continue
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())  # NaN comparisons are False
        results.append(_result("range", col, ERROR if bad else OK, bad, n, f"allowed [{lo}, {hi}]"))
    return results


def check_categories(df: pd.DataFrame, allowed: dict) -> list:
    results = []
    n = len(df)
    for col, values in allowed.items():
        if col not in df.columns:
            continue
        bad_mask = df[col].notna() & ~df[col].isin(values)
        bad = int(bad_mask.sum())
        shown = sorted(df.loc[bad_mask, col].astype(str).unique().tolist())
        msg = f"allowed {values}" + (f"; found invalid {shown}" if bad else "")
        results.append(_result("categories", col, WARN if bad else OK, bad, n, msg))
    return results


def check_target(df: pd.DataFrame, target: str) -> list:
    n = len(df)
    bad = int((~df[target].isin([0, 1])).sum())
    rate = df[target].mean()
    return [
        _result("target_values", target, ERROR if bad else OK, bad, n, "must be 0 or 1"),
        _result("target_rate", target, OK, 0, n, f"positive rate = {rate:.2%}"),
    ]


def check_dates(df: pd.DataFrame, col: str = "snapshot_date") -> list:
    parsed = pd.to_datetime(df[col], errors="coerce")
    bad = int(parsed.isna().sum())
    return [_result("date_format", col, ERROR if bad else OK, bad, len(df), "must be parseable dates")]


def validate(df: pd.DataFrame, cfg: dict) -> list:
    v = cfg["validation"]
    results = []
    results += check_schema(df, v["expected_columns"])
    results += check_duplicates(df, v["id_column"])
    results += check_missing(df, v["max_missing_fraction"])
    results += check_ranges(df, v["ranges"])
    results += check_categories(df, v["allowed_values"])
    results += check_target(df, v["target_column"])
    results += check_dates(df)
    return results


def print_report(results: list) -> None:
    tag = {ERROR: "[FAIL]", WARN: "[WARN]", OK: "[ OK ]"}
    for r in results:
        print(f"{tag[r['severity']]} {r['check']:<15} {r['column']:<28} "
              f"failed={r['failed_rows']:>6} ({r['failed_pct']:>6.2f}%)  {r['message']}")
    n_err = sum(r["severity"] == ERROR for r in results)
    n_warn = sum(r["severity"] == WARN for r in results)
    print(f"\nSummary: {n_err} errors, {n_warn} warnings, {len(results)} checks")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit with code 1 if any ERROR")
    args = parser.parse_args()

    cfg = load_config()
    df = pd.read_csv(ROOT / cfg["paths"]["raw_data"])
    print(f"Validating {len(df):,} rows x {df.shape[1]} columns\n")

    results = validate(df, cfg)
    print_report(results)

    report_path = ROOT / cfg["paths"]["reports_dir"] / "data_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report saved -> {report_path}")

    has_errors = any(r["severity"] == ERROR for r in results)
    return 1 if (args.strict and has_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
