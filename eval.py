"""Evaluation framework for mera.health lab report extractor.

Usage:
    python eval.py                      # Evaluates current extract.py against gold labels
    python eval.py --extractor extract   # Specify module name
    python eval.py --output baseline.json

Scores:
    - Strict Row Match (Name, Value, Unit, Reference Range all match)
    - Clinical Value Match (Name and Value match)
    - Field-level Precision, Recall, and F1
    - Latency and estimated API cost
"""
import argparse
import difflib
import importlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# Gemini 2.5/3.8 Flash approximate pricing per million tokens
INPUT_COST_PER_M = 0.15
OUTPUT_COST_PER_M = 0.60


def normalize_text(s) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    # Normalize greek letters/micro signs
    s = s.replace("µ", "u").replace("μ", "u")
    # Replace dashes/spaces
    s = re.sub(r"[\s\-_/]+", " ", s)
    return s.strip()


def normalize_unit(u) -> str:
    if u is None:
        return ""
    u = str(u).strip().lower()
    u = u.replace("µ", "u").replace("μ", "u")
    u = re.sub(r"\s+", "", u)
    return u


def normalize_range(r) -> str:
    if r is None:
        return ""
    r = str(r).strip().lower()
    r = r.replace("µ", "u").replace("μ", "u")
    # normalize spaces around hyphens and operators
    r = re.sub(r"\s*-\s*", "-", r)
    r = re.sub(r"\s*to\s*", "-", r)
    r = re.sub(r"\s*<\s*", "<", r)
    r = re.sub(r"\s*>\s*", ">", r)
    r = re.sub(r"\s+", " ", r)
    return r.strip()


def parse_numeric(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    v_str = str(v).strip().replace(",", "")
    # Check simple float
    try:
        return float(v_str)
    except ValueError:
        return None


def values_match(pred, gold) -> bool:
    if pred is None and gold is None:
        return True
    if pred is None or gold is None:
        return False

    p_num = parse_numeric(pred)
    g_num = parse_numeric(gold)

    if p_num is not None and g_num is not None:
        if math.isclose(p_num, g_num, rel_tol=1e-3, abs_tol=1e-4):
            return True
        return False

    # String comparison (case-insensitive, whitespace normalized)
    p_norm = normalize_text(pred)
    g_norm = normalize_text(gold)
    if p_norm == g_norm:
        return True

    # Check for presence of key terms like positive/present/negative/nil
    qualitative_terms = ["negative", "positive", "present", "absent", "clear", "nil", "normal"]
    for term in qualitative_terms:
        if term in p_norm and term in g_norm:
            return True

    return False


def test_name_similarity(p_name: str, g_name: str) -> float:
    p_norm = normalize_text(p_name)
    g_norm = normalize_text(g_name)
    if p_norm == g_norm:
        return 1.0

    # Token overlap
    p_tokens = set(p_norm.split())
    g_tokens = set(g_norm.split())
    if p_tokens and g_tokens:
        jaccard = len(p_tokens & g_tokens) / len(p_tokens | g_tokens)
        if jaccard >= 0.7:
            return 0.95

    # Sequence matcher ratio
    return difflib.SequenceMatcher(None, p_norm, g_norm).ratio()


def match_tests(pred_tests: list[dict], gold_tests: list[dict]):
    """Match predicted tests to gold tests using greedy maximum name similarity."""
    pairs = []
    for p_idx, p in enumerate(pred_tests):
        for g_idx, g in enumerate(gold_tests):
            sim = test_name_similarity(p.get("test_name", ""), g.get("test_name", ""))
            if sim >= 0.65:
                pairs.append((sim, p_idx, g_idx))

    # Sort descending by similarity
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gold = set()
    matches = []

    for sim, p_idx, g_idx in pairs:
        if p_idx not in matched_pred and g_idx not in matched_gold:
            matched_pred.add(p_idx)
            matched_gold.add(g_idx)
            matches.append((p_idx, g_idx, sim))

    unmatched_pred = [p_idx for p_idx in range(len(pred_tests)) if p_idx not in matched_pred]
    unmatched_gold = [g_idx for g_idx in range(len(gold_tests)) if g_idx not in matched_gold]

    return matches, unmatched_pred, unmatched_gold


def evaluate_report(pred_tests: list[dict], gold_tests: list[dict]) -> dict:
    matches, unmatched_pred, unmatched_gold = match_tests(pred_tests, gold_tests)

    strict_matches = 0
    value_matches = 0
    unit_matches = 0
    range_matches = 0
    name_matches = len(matches)

    details = []

    for p_idx, g_idx, sim in matches:
        p = pred_tests[p_idx]
        g = gold_tests[g_idx]

        v_ok = values_match(p.get("value"), g.get("value"))
        u_ok = normalize_unit(p.get("unit")) == normalize_unit(g.get("unit"))
        r_ok = normalize_range(p.get("reference_range")) == normalize_range(g.get("reference_range"))
        strict_ok = v_ok and u_ok and r_ok

        if strict_ok:
            strict_matches += 1
        if v_ok:
            value_matches += 1
        if u_ok:
            unit_matches += 1
        if r_ok:
            range_matches += 1

        details.append({
            "pred_name": p.get("test_name"),
            "gold_name": g.get("test_name"),
            "name_sim": round(sim, 2),
            "pred_value": p.get("value"),
            "gold_value": g.get("value"),
            "value_ok": v_ok,
            "pred_unit": p.get("unit"),
            "gold_unit": g.get("unit"),
            "unit_ok": u_ok,
            "pred_range": p.get("reference_range"),
            "gold_range": g.get("reference_range"),
            "range_ok": r_ok,
            "strict_ok": strict_ok,
        })

    n_gold = len(gold_tests)
    n_pred = len(pred_tests)

    # Precision, Recall, F1 for Strict Match
    strict_p = strict_matches / n_pred if n_pred > 0 else 0.0
    strict_r = strict_matches / n_gold if n_gold > 0 else 0.0
    strict_f1 = (2 * strict_p * strict_r / (strict_p + strict_r)) if (strict_p + strict_r) > 0 else 0.0

    # Precision, Recall, F1 for Clinical Value Match
    val_p = value_matches / n_pred if n_pred > 0 else 0.0
    val_r = value_matches / n_gold if n_gold > 0 else 0.0
    val_f1 = (2 * val_p * val_r / (val_p + val_r)) if (val_p + val_r) > 0 else 0.0

    return {
        "gold_count": n_gold,
        "pred_count": n_pred,
        "name_matches": name_matches,
        "value_matches": value_matches,
        "unit_matches": unit_matches,
        "range_matches": range_matches,
        "strict_matches": strict_matches,
        "strict_precision": round(strict_p, 4),
        "strict_recall": round(strict_r, 4),
        "strict_f1": round(strict_f1, 4),
        "value_precision": round(val_p, 4),
        "value_recall": round(val_r, 4),
        "value_f1": round(val_f1, 4),
        "details": details,
        "unmatched_gold": [gold_tests[i].get("test_name") for i in unmatched_gold],
        "unmatched_pred": [pred_tests[i].get("test_name") for i in unmatched_pred],
    }


def run_eval(extractor_module_name: str = "extract", output_file: str = None):
    # Ensure local poppler binary directory is in PATH on Windows
    poppler_bin = Path("poppler/poppler-24.08.0/Library/bin")
    if poppler_bin.exists():
        os.environ["PATH"] = f"{poppler_bin.resolve()};{os.environ.get('PATH', '')}"

    # Default model fallback if not set
    if "GEMINI_MODEL" not in os.environ:
        os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"

    extractor = importlib.import_module(extractor_module_name)

    gold_dir = Path("gold")
    reports_dir = Path("reports")

    gold_files = sorted(gold_dir.glob("*.json"))
    if not gold_files:
        print("No gold files found in gold/")
        return

    print("=" * 80)
    print(f"EVALUATING: {extractor_module_name}.py (Model: {os.environ.get('GEMINI_MODEL')})")
    print(f"Total labelled reports: {len(gold_files)}")
    print("=" * 80)

    report_results = {}
    total_gold = 0
    total_pred = 0
    total_strict = 0
    total_val = 0
    total_name = 0
    total_latency = 0.0

    # Table Header
    print(f"{'Report':<15} | {'Gold':>4} | {'Pred':>4} | {'NameMatch':>9} | {'ValMatch':>8} | {'Strict':>6} | {'Val F1':>7} | {'Strict F1':>9} | {'Latency':>7}")
    print("-" * 88)

    for g_path in gold_files:
        report_id = g_path.stem  # e.g. report_01
        gold_data = json.loads(g_path.read_text(encoding="utf-8")).get("tests", [])

        # Find matching report file
        candidates = list(reports_dir.glob(f"{report_id}.*"))
        if not candidates:
            print(f"Report document for {report_id} not found in reports/")
            continue
        report_path = candidates[0]

        start_time = time.time()
        # Call extractor
        try:
            pred_output = extractor.extract(report_path)
            pred_data = pred_output.get("tests", []) if isinstance(pred_output, dict) else []
        except Exception as e:
            print(f"Error extracting {report_path}: {e}")
            pred_data = []
        elapsed = time.time() - start_time
        total_latency += elapsed

        eval_res = evaluate_report(pred_data, gold_data)
        eval_res["latency_seconds"] = round(elapsed, 2)
        report_results[report_id] = eval_res

        total_gold += eval_res["gold_count"]
        total_pred += eval_res["pred_count"]
        total_strict += eval_res["strict_matches"]
        total_val += eval_res["value_matches"]
        total_name += eval_res["name_matches"]

        print(f"{report_id:<15} | {eval_res['gold_count']:>4} | {eval_res['pred_count']:>4} | {eval_res['name_matches']:>9} | {eval_res['value_matches']:>8} | {eval_res['strict_matches']:>6} | {eval_res['value_f1']:>7.3f} | {eval_res['strict_f1']:>9.3f} | {elapsed:>6.2f}s")

    print("-" * 88)

    # Micro-averages across all tests
    strict_p = total_strict / total_pred if total_pred > 0 else 0.0
    strict_r = total_strict / total_gold if total_gold > 0 else 0.0
    strict_f1 = (2 * strict_p * strict_r / (strict_p + strict_r)) if (strict_p + strict_r) > 0 else 0.0

    val_p = total_val / total_pred if total_pred > 0 else 0.0
    val_r = total_val / total_gold if total_gold > 0 else 0.0
    val_f1 = (2 * val_p * val_r / (val_p + val_r)) if (val_p + val_r) > 0 else 0.0

    name_recall = total_name / total_gold if total_gold > 0 else 0.0
    avg_latency = total_latency / len(gold_files) if gold_files else 0.0

    # Approximate Cost calculation based on token estimation
    # Flash: approx 1500 tokens input per image page, ~400 tokens output
    estimated_cost_per_report = ((1500 / 1e6) * INPUT_COST_PER_M) + ((400 / 1e6) * OUTPUT_COST_PER_M)

    summary = {
        "reports_evaluated": len(gold_files),
        "total_gold_tests": total_gold,
        "total_pred_tests": total_pred,
        "test_name_recall": round(name_recall, 4),
        "clinical_value": {
            "matches": total_val,
            "precision": round(val_p, 4),
            "recall": round(val_r, 4),
            "f1": round(val_f1, 4),
        },
        "strict_overall": {
            "matches": total_strict,
            "precision": round(strict_p, 4),
            "recall": round(strict_r, 4),
            "f1": round(strict_f1, 4),
        },
        "avg_latency_seconds": round(avg_latency, 2),
        "total_latency_seconds": round(total_latency, 2),
        "estimated_cost_usd_per_report": round(estimated_cost_per_report, 5),
    }

    print("\nOVERALL SUMMARY:")
    print(f"  Total Gold Tests:           {total_gold}")
    print(f"  Total Extracted Tests:      {total_pred}")
    print(f"  Test Name Recall:           {name_recall * 100:.1f}%")
    print(f"  Clinical Value Match F1:    {val_f1 * 100:.1f}% (P: {val_p*100:.1f}%, R: {val_r*100:.1f}%)")
    print(f"  Strict Row Match F1:        {strict_f1 * 100:.1f}% (P: {strict_p*100:.1f}%, R: {strict_r*100:.1f}%)")
    print(f"  Average Latency:            {avg_latency:.2f}s per report")
    print(f"  Estimated Cost:             ${estimated_cost_per_report:.5f} / report (~${estimated_cost_per_report*10000:.2f} per 10k reports)")
    print("=" * 80)

    if output_file:
        out_path = Path(output_file)
        payload = {"summary": summary, "reports": report_results}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved evaluation results to {output_file}")

    return summary, report_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate lab report extraction.")
    parser.add_argument("--extractor", default="extract", help="Module name of extractor")
    parser.add_argument("--output", default=None, help="Path to save output JSON")
    args = parser.parse_args()

    run_eval(extractor_module_name=args.extractor, output_file=args.output)
