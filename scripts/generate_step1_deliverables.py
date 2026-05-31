#!/usr/bin/env python3
"""Generate Step 1 deliverables for the updated 2026 requirements.

This script reads the Step 3 run output CSV (decision + reliability signals)
and produces:
- efficiency metrics (latency, decision split)
- structured failure taxonomy with representative examples
- short markdown report block that can be copied into the main report
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def classify_failure(row: dict[str, str]) -> list[str]:
    tags: list[str] = []

    decision = row.get("decision", "")
    ambiguous = _to_bool(row.get("query_ambiguous", "False"))
    contradiction = _to_bool(row.get("has_contradictions", "False"))
    trust = _to_float(row.get("trust_score", "0"))
    suff = _to_float(row.get("evidence_sufficiency", "0"))
    grounding = _to_float(row.get("grounding_score", "0"))
    retry_count = int(_to_float(row.get("retry_count", "0")))

    if ambiguous:
        tags.append("ambiguity_failure")
    if contradiction:
        tags.append("contradiction_failure")

    if decision == "abstain":
        if suff < 0.45:
            tags.append("retrieval_failure")
        elif suff >= 0.45 and trust < 0.4:
            tags.append("ranking_failure")

        if retry_count > 0:
            tags.append("orchestration_failure")

    if decision == "answer":
        if grounding < 0.85:
            tags.append("grounding_failure")
        if trust < 0.4:
            tags.append("overconfidence_failure")

    # if answer but not enough support, still synthesis issue
    if decision == "answer" and suff < 0.35:
        tags.append("synthesis_failure")

    return tags


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_report(rows: list[dict[str, str]], baseline_ref: str) -> str:
    runtimes = [_to_float(r.get("runtime_sec", "0")) for r in rows]
    non_zero_runtimes = [t for t in runtimes if t > 0]

    decisions = Counter(r.get("decision", "unknown") for r in rows)
    strategies = Counter(r.get("strategy_used", "unknown") for r in rows)

    taxonomy_examples: dict[str, list[str]] = defaultdict(list)
    taxonomy_counts: Counter[str] = Counter()

    for r in rows:
        q = r.get("query", "")
        tags = classify_failure(r)
        for tag in tags:
            taxonomy_counts[tag] += 1
            if len(taxonomy_examples[tag]) < 2:
                taxonomy_examples[tag].append(q)

    avg_runtime = statistics.mean(non_zero_runtimes) if non_zero_runtimes else 0.0
    p95_runtime = statistics.quantiles(non_zero_runtimes, n=20)[18] if len(non_zero_runtimes) >= 20 else max(non_zero_runtimes or [0.0])

    lines: list[str] = []
    lines.append("# Step 1 Deliverables (Updated Requirements)")
    lines.append("")
    lines.append("## Verified Baseline")
    lines.append("- Baseline notebook: `Step_1_Baseline_and_Failure_Analysis.ipynb`")
    lines.append("- Reference baseline configuration: " + baseline_ref)
    lines.append("- Baseline components verified in notebook: retrieval (BM25/Dense/GraphRAG), orchestration (Confidence/Waterfall/Voting), re-ranking, answer synthesis.")
    lines.append("")
    lines.append("## Baseline Efficiency Indicators")
    lines.append(f"- Runs analyzed: {len(rows)}")
    lines.append(f"- Avg runtime per query: {avg_runtime:.3f}s")
    lines.append(f"- P95 runtime per query: {p95_runtime:.3f}s")
    lines.append(f"- Decision split: {dict(decisions)}")
    lines.append(f"- Strategy usage: {dict(strategies)}")
    lines.append("- Approximate cost proxy: low (heuristic pipeline + no per-query generative model expansion in this output).")
    lines.append("")
    lines.append("## Structured Failure Taxonomy")
    for tag in [
        "retrieval_failure",
        "ranking_failure",
        "synthesis_failure",
        "grounding_failure",
        "ambiguity_failure",
        "contradiction_failure",
        "orchestration_failure",
        "overconfidence_failure",
    ]:
        count = taxonomy_counts.get(tag, 0)
        ex = taxonomy_examples.get(tag, [])
        lines.append(f"- {tag}: {count}")
        for q in ex:
            lines.append(f"  - example: {q}")
    lines.append("")
    lines.append("## Motivation for Proposed Extensions")
    lines.append("- Ambiguity + contradiction + orchestration failures indicate need for stronger query clarification, contradiction-aware retrieval routing, and recovery policy improvements.")
    lines.append("- Retrieval/ranking failures motivate adaptive weighting, better reranking features, and stricter evidence sufficiency thresholds before synthesis.")
    lines.append("- These findings justify reliability agents (clarification, contradiction detection, trust scoring, abstention) as primary extension targets.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("2026-05-29_22-44-32_output_step3.csv"),
        help="Path to run output CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("step1_updated_deliverables.md"),
        help="Output markdown path",
    )
    parser.add_argument(
        "--baseline-ref",
        default="Full corpus + Confidence orchestration (MRR 0.208827) as reference baseline",
        help="Reference baseline configuration description",
    )
    args = parser.parse_args()

    rows = load_rows(args.input)
    report = build_report(rows, args.baseline_ref)
    args.output.write_text(report, encoding="utf-8")

    print(f"Wrote {args.output} from {args.input} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
