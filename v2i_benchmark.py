"""Phase 10: V2I Smart Intersection Benchmark CLI.

Executes controlled paired trials comparing BASELINE_NO_V2I vs V2I_ENABLED.
Generates human-readable console report and machine-readable JSON & CSV exports.

Usage:
    python v2i_benchmark.py [--trials N] [--json path] [--csv path]
"""

import argparse
import os
import sys

# Ensure headless pygame execution when invoked in console
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
pygame.init()

from v2i.metrics import ExperimentSuite, ComparisonReport


def main():
    parser = argparse.ArgumentParser(description="V2I Smart Intersection Quantitative Benchmark")
    parser.add_argument("--trials", type=int, default=10, help="Number of paired trials to execute (default: 10)")
    parser.add_argument("--json", type=str, default="benchmark_results.json", help="Path for JSON output")
    parser.add_argument("--csv", type=str, default="benchmark_results.csv", help="Path for CSV output")
    args = parser.parse_args()

    print(f"\n=================================================================")
    print(f"  V2I SMART INTERSECTION — PHASE 10 QUANTITATIVE BENCHMARK       ")
    print(f"=================================================================")
    print(f"Running {args.trials} controlled paired trials (BASELINE vs V2I)...")
    print(f"Please wait while deterministic simulations execute...\n")

    configs = ExperimentSuite.generate_default_configs(num_trials=args.trials)
    base_results, v2i_results = ExperimentSuite.run_paired_suite(configs=configs, num_trials=args.trials)

    report = ComparisonReport(base_results, v2i_results)
    text_report = report.generate_text_report()
    print("\n" + text_report + "\n")

    if args.json:
        report.export_json(args.json)
        print(f"[BENCHMARK] Exported machine-readable results to: {args.json}")

    if args.csv:
        report.export_csv(args.csv)
        print(f"[BENCHMARK] Exported tabular comparison CSV to: {args.csv}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
