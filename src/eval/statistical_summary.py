"""Statistical summary across multiple experiment runs.

Computes mean ± std for SVR, SCR, EA (and per-difficulty breakdowns)
from multiple summary JSON files grouped by (model, setting).

Usage:
    python -m src.eval.statistical_summary --paths results/exp1_summary.json results/exp2_summary.json ...

    # Or use globs:
    python -m src.eval.statistical_summary --paths results/experiment_20260324_*_cypher_soft_summary.json

    # Filter by setting (S1/S2/S3) or model:
    python -m src.eval.statistical_summary --paths results/*.json --settings S1 S2 S3
    python -m src.eval.statistical_summary --paths results/*.json --models "gemini-2.5-flash"

    # Output LaTeX table rows:
    python -m src.eval.statistical_summary --paths results/*.json --latex
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import math

SETTING_MAP = {
    "direct_qa_baseline": "S1 (Naive QA)",
    "direct_qa_grounded": "S2 (Grounded QA)",
    "cypher_soft": "S3 (Soft Cypher)",
    "cypher_strict": "S4 (Strict Cypher)",
}

SETTING_ALIASES = {
    "S1": "direct_qa_baseline",
    "S2": "direct_qa_grounded",
    "S3": "cypher_soft",
    "S4": "cypher_strict",
}

DIFFICULTIES = ["easy", "medium", "hard"]

KEY_METRICS = ["svr_mean", "scr_mean", "ea_mean"]
KEY_METRIC_LABELS = {"svr_mean": "SVR", "scr_mean": "SCR", "ea_mean": "EA"}


def load_summary(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
        if "config" not in data or "metrics" not in data:
            return None
        if "setting" not in data["config"] or "model" not in data["config"]:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1)."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def fmt_stat(values: list[float], fmt: str = ".2f") -> str:
    """Format mean ± std."""
    mean = _mean(values)
    std = _std(values)
    if std == 0.0:
        return f"{mean:{fmt}}"
    return f"{mean:{fmt}} ± {std:{fmt}}"


def fmt_stat_latex(values: list[float], fmt: str = ".2f") -> str:
    """Format mean ± std for LaTeX."""
    mean = _mean(values)
    std = _std(values)
    if std == 0.0:
        return f"{mean:{fmt}}"
    return f"${mean:{fmt}} \\pm {std:{fmt}}$"


def print_group_stats(
    runs: list[dict],
    model: str,
    setting: str,
    latex: bool = False,
):
    n = len(runs)
    setting_label = SETTING_MAP.get(setting, setting)

    print(f"\n{'='*70}")
    print(f"  Model: {model}")
    print(f"  Setting: {setting_label}")
    print(f"  Runs: {n}")
    print(f"{'='*70}")

    if n < 2:
        print("  ⚠  Only 1 run — no std can be computed.")

    # Overall metrics
    print(f"\n  {'Metric':<8} {'Values across runs':<40} {'Mean ± Std':<20}")
    print(f"  {'-'*68}")
    overall_stats = {}
    for metric in KEY_METRICS:
        values = [r["metrics"][metric] for r in runs]
        label = KEY_METRIC_LABELS[metric]
        val_str = ", ".join(f"{v:.4f}" for v in values)
        overall_stats[metric] = values
        print(f"  {label:<8} {val_str:<40} {fmt_stat(values)}")

    # Per-difficulty
    print(f"\n  Per-difficulty breakdown:")
    for diff in DIFFICULTIES:
        print(f"\n  [{diff.capitalize()}]")
        print(f"  {'Metric':<8} {'Values across runs':<40} {'Mean ± Std':<20}")
        print(f"  {'-'*68}")
        for metric in KEY_METRICS:
            values = []
            for r in runs:
                by_diff = r["metrics"].get("metrics_by_difficulty", {})
                if diff in by_diff:
                    values.append(by_diff[diff].get(metric, float("nan")))
            if not values:
                continue
            label = KEY_METRIC_LABELS[metric]
            val_str = ", ".join(f"{v:.4f}" for v in values)
            print(f"  {label:<8} {val_str:<40} {fmt_stat(values)}")

    # LaTeX row
    if latex:
        svr_tex = fmt_stat_latex(overall_stats["svr_mean"])
        scr_tex = fmt_stat_latex(overall_stats["scr_mean"])
        ea_tex = fmt_stat_latex(overall_stats["ea_mean"])
        print(f"\n  LaTeX row:")
        print(f"  & {setting_label} & {svr_tex} & {scr_tex} & {ea_tex} \\\\")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical summary across multiple experiment runs."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Paths to summary JSON files.",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        default=None,
        help="Filter by setting aliases (e.g., S1 S2 S3). Default: all.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Filter by model name substring (e.g., 'gemini-2.5').",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Output LaTeX-formatted table rows.",
    )
    args = parser.parse_args()

    # Resolve setting filters
    setting_filter = None
    if args.settings:
        setting_filter = set()
        for s in args.settings:
            s_upper = s.upper()
            if s_upper in SETTING_ALIASES:
                setting_filter.add(SETTING_ALIASES[s_upper])
            else:
                setting_filter.add(s)

    # Load all summaries
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped = 0

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} does not exist, skipping.", file=sys.stderr)
            skipped += 1
            continue

        data = load_summary(path)
        if data is None:
            skipped += 1
            continue

        model = data["config"]["model"]
        setting = data["config"]["setting"]

        # Apply filters
        if setting_filter and setting not in setting_filter:
            continue
        if args.models:
            if not any(m.lower() in model.lower() for m in args.models):
                continue

        groups[(model, setting)].append(data)

    if not groups:
        print("No matching summary files found.", file=sys.stderr)
        sys.exit(1)

    # Sort groups by model then setting
    setting_order = list(SETTING_MAP.keys())
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (k[0], setting_order.index(k[1]) if k[1] in setting_order else 99),
    )

    print(f"Loaded {sum(len(v) for v in groups.values())} runs across {len(groups)} (model, setting) groups.")
    if skipped:
        print(f"Skipped {skipped} files (not valid summaries).")

    for key in sorted_keys:
        model, setting = key
        runs = groups[key]
        print_group_stats(runs, model, setting, latex=args.latex)

    # Summary table at the end
    if args.latex:
        print(f"\n{'='*70}")
        print("  Complete LaTeX table rows (copy-paste ready):")
        print(f"{'='*70}")
        current_model = None
        for key in sorted_keys:
            model, setting = key
            runs = groups[key]
            setting_label = SETTING_MAP.get(setting, setting)
            svr = fmt_stat_latex([r["metrics"]["svr_mean"] for r in runs])
            scr = fmt_stat_latex([r["metrics"]["scr_mean"] for r in runs])
            ea = fmt_stat_latex([r["metrics"]["ea_mean"] for r in runs])
            if model != current_model:
                if current_model is not None:
                    print("\\hline")
                print(f"\\textit{{{model}}}")
                current_model = model
            print(f"  & {setting_label} & {svr} & {scr} & {ea} \\\\")
        print("\\hline")


if __name__ == "__main__":
    main()
