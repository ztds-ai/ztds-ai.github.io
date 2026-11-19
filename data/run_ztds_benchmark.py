"""Reproducible ZTDS Reference Engine Micro-Benchmark (RFC v1.0).

Measures local volatile-RAM sanitization + bijective restoration latency
(excluding explicit teardown gc.collect()) as well as full session lifecycle
latency with Theorem 2 RAM zeroization across a 100% synthetic enterprise corpus
(4 payload classes, 40 documents).

All inputs are fabricated RFC 2606 / standard test identifiers.
Zero external network calls are performed.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

import ztds


BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "synthetic_pii_corpus.csv"
RESULTS_CSV_PATH = BASE_DIR / "benchmark_results.csv"
RESULTS_JSON_PATH = BASE_DIR / "benchmark_results.json"

ITERATIONS_PER_CLASS = 500


def percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def load_corpus() -> Dict[str, List[Dict[str, str]]]:
    by_class: Dict[str, List[Dict[str, str]]] = {}
    with CORPUS_PATH.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            by_class.setdefault(row["payload_class"], []).append(row)
    return by_class


def run() -> None:
    corpus = load_corpus()
    summary_rows: List[Dict[str, object]] = []

    with ztds.session() as warm:
        warm.sanitize("Warmup alice@example.invalid 2026-01-15 DE89370400440532013000")

    for payload_class, docs in corpus.items():
        sanitize_reveal_ms: List[float] = []
        full_cycle_ms: List[float] = []
        exact_matches = 0
        total_bytes = 0
        total_tokens = 0

        tracemalloc.start()
        for i in range(ITERATIONS_PER_CLASS):
            doc = docs[i % len(docs)]
            payload = doc["payload_text"]
            total_bytes += len(payload.encode("utf-8"))

            fc0 = time.perf_counter_ns()
            with ztds.session() as s:
                sr0 = time.perf_counter_ns()
                sanitized, minted = s.sanitize(payload)
                restored = s.reveal(sanitized)
                sr1 = time.perf_counter_ns()
            fc1 = time.perf_counter_ns()

            sanitize_reveal_ms.append((sr1 - sr0) / 1_000_000.0)
            full_cycle_ms.append((fc1 - fc0) / 1_000_000.0)
            total_tokens += len(minted)
            if restored == payload and len(minted) > 0:
                exact_matches += 1

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        sanitize_reveal_ms.sort()
        full_cycle_ms.sort()
        total_sr_sec = sum(sanitize_reveal_ms) / 1000.0
        avg_bytes = total_bytes / ITERATIONS_PER_CLASS

        row = {
            "payload_class": payload_class,
            "unique_templates": len(docs),
            "iterations": ITERATIONS_PER_CLASS,
            "avg_payload_bytes": round(avg_bytes, 1),
            "mean_tokens_per_doc": round(total_tokens / ITERATIONS_PER_CLASS, 2),
            "sanitize_reveal_p50_ms": round(percentile(sanitize_reveal_ms, 50), 4),
            "sanitize_reveal_mean_ms": round(statistics.mean(sanitize_reveal_ms), 4),
            "sanitize_reveal_p95_ms": round(percentile(sanitize_reveal_ms, 95), 4),
            "sanitize_reveal_p99_ms": round(percentile(sanitize_reveal_ms, 99), 4),
            "full_cycle_with_gc_zeroize_p50_ms": round(percentile(full_cycle_ms, 50), 4),
            "throughput_mb_s": round((total_bytes / (1024 * 1024)) / total_sr_sec, 2) if total_sr_sec > 0 else 0.0,
            "round_trip_exact_match_pct": round((exact_matches / ITERATIONS_PER_CLASS) * 100.0, 2),
            "peak_tracemalloc_kb": round(peak_bytes / 1024.0, 2),
        }
        summary_rows.append(row)

    fieldnames = list(summary_rows[0].keys())
    with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    output_doc = {
        "ztds_version": getattr(ztds, "__version__", "unknown"),
        "spec_reference": "IETF draft-sibiryakov-ztds-protocol-02 (RFC v1.0)",
        "iterations_per_class": ITERATIONS_PER_CLASS,
        "total_iterations": ITERATIONS_PER_CLASS * len(summary_rows),
        "results": summary_rows,
    }
    RESULTS_JSON_PATH.write_text(json.dumps(output_doc, indent=2), encoding="utf-8")
    print(json.dumps(output_doc, indent=2))


if __name__ == "__main__":
    run()
