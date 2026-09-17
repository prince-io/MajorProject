"""Generate paper-ready figures from training logs and evaluation reports.

Produces, per experiment (``results/experiments/<exp>/figures/``):

- ``training_curves.png`` - loss components and mAP over epochs (from ``results.csv``)
- ``bars_<report>.png``   - overall + per-weather + per-class mAP50 bars

and cross-experiment comparisons (``results/summary/figures/``):

- ``comparison_overall.png``    - overall mAP50 per experiment (error bars from folds)
- ``comparison_per_weather.png``- per-weather mAP50 grouped by experiment
- ``comparison_per_class.png``  - per-class mAP50 grouped by experiment

Usage::

    python src/visualize.py            # all experiments + summary
    python src/visualize.py --exp B0_bdd_src
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ACDC_WEATHERS,
    EXPERIMENTS_DIR,
    SUMMARY_DIR,
    UNIFIED_CLASSES,
    ensure_dir,
)

SUMMARY_PATH = SUMMARY_DIR / "summary.json"
CLASS_ORDER = [UNIFIED_CLASSES[i] for i in sorted(UNIFIED_CLASSES)]


def _read_csv(path: Path) -> dict[str, list[float]]:
    """Read an Ultralytics results.csv into column -> list of floats."""
    columns: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                key = key.strip()
                try:
                    columns.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    continue
    return columns


def _find(columns: dict[str, list[float]], needle: str) -> list[float] | None:
    """Return the first column whose name contains ``needle``."""
    for key, values in columns.items():
        if needle in key:
            return values
    return None


def plot_training_curves(csv_path: Path, out_png: Path) -> bool:
    """Plot loss and mAP curves from a results.csv. Returns True if written."""
    columns = _read_csv(csv_path)
    if not columns:
        return False
    fig, (ax_loss, ax_map) = plt.subplots(1, 2, figsize=(12, 4.5))

    for needle, label in (("box_loss", "box"), ("cls_loss", "cls"), ("dfl_loss", "dfl")):
        values = _find(columns, needle)
        if values:
            ax_loss.plot(values, label=label)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training losses")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    for needle, label in (("mAP50(B)", "mAP50"), ("mAP50-95(B)", "mAP50-95")):
        values = _find(columns, needle)
        if values:
            ax_map.plot(values, label=label)
    ax_map.set_xlabel("epoch")
    ax_map.set_ylabel("mAP")
    ax_map.set_ylim(0, 1)
    ax_map.set_title("Validation mAP")
    ax_map.legend()
    ax_map.grid(alpha=0.3)

    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def plot_experiment_bars(report: dict, out_png: Path) -> bool:
    """Plot overall + per-weather + per-class mAP50 bars for one eval report."""
    overall = report.get("overall", {})
    per_weather = report.get("per_weather", {})
    per_class = overall.get("per_class", {})

    fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(13, 4.5))

    weather_labels = [w for w in ACDC_WEATHERS if w in per_weather]
    ax_w.bar(["overall", *weather_labels], [overall.get("mAP50", 0.0)]
             + [per_weather[w].get("mAP50", 0.0) for w in weather_labels], color="#3b6ea5")
    ax_w.set_ylim(0, 1)
    ax_w.set_ylabel("mAP50")
    ax_w.set_title(f"Overall and per-weather mAP50 — {report.get('run', '')}")
    ax_w.grid(axis="y", alpha=0.3)

    class_labels = [c for c in CLASS_ORDER if c in per_class]
    ax_c.bar(class_labels, [per_class[c].get("mAP50", 0.0) for c in class_labels], color="#a5603b")
    ax_c.set_ylim(0, 1)
    ax_c.set_ylabel("mAP50")
    ax_c.set_title("Per-class mAP50")
    ax_c.grid(axis="y", alpha=0.3)

    ensure_dir(out_png.parent)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def _collect(summary: dict) -> dict[str, dict]:
    """Extract (mean, std) mAP50 per experiment from the summary."""
    collected: dict[str, dict] = {}
    for exp, entry in summary.items():
        rec: dict = {"overall": (None, None), "weather": {}, "class": {}}
        if "fold_agg" in entry:
            overall = entry["fold_agg"]["overall"]["metrics"].get("mAP50", {})
            rec["overall"] = (overall.get("mean"), overall.get("std"))
            for weather, block in entry["fold_agg"].get("per_weather", {}).items():
                metric = block["metrics"].get("mAP50", {})
                rec["weather"][weather] = (metric.get("mean"), metric.get("std"))
        elif "official" in entry:
            rec["overall"] = (entry["official"]["overall"].get("mAP50"), None)
            for weather, block in entry["official"].get("per_weather", {}).items():
                rec["weather"][weather] = (block.get("mAP50"), None)
        elif "single" in entry:
            rec["overall"] = (entry["single"]["overall"].get("mAP50"), None)
            for weather, block in entry["single"].get("per_weather", {}).items():
                rec["weather"][weather] = (block.get("mAP50"), None)
        collected[exp] = rec
    return collected


def plot_comparison(summary: dict, out_dir: Path) -> None:
    """Write overall, per-weather, and per-class comparison figures."""
    data = _collect(summary)
    if not data:
        return
    ensure_dir(out_dir)
    experiments = list(data)
    x = range(len(experiments))

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(experiments)), 4.5))
    means = [data[e]["overall"][0] or 0.0 for e in experiments]
    stds = [data[e]["overall"][1] or 0.0 for e in experiments]
    ax.bar(x, means, yerr=stds, capsize=4, color="#3b6ea5")
    ax.set_xticks(list(x))
    ax.set_xticklabels(experiments, rotation=30, ha="right")
    ax.set_ylabel("mAP50")
    ax.set_ylim(0, 1)
    ax.set_title("Overall mAP50 by experiment")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_overall.png", dpi=150)
    plt.close(fig)

    width = 0.8 / max(1, len(experiments))
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = list(range(len(ACDC_WEATHERS)))
    for i, exp in enumerate(experiments):
        offsets = [p + i * width - 0.4 + width / 2 for p in positions]
        values = [data[exp]["weather"].get(w, (0.0, None))[0] or 0.0 for w in ACDC_WEATHERS]
        errors = [data[exp]["weather"].get(w, (None, 0.0))[1] or 0.0 for w in ACDC_WEATHERS]
        ax.bar(offsets, values, width=width, yerr=errors, capsize=3, label=exp)
    ax.set_xticks(positions)
    ax.set_xticklabels(ACDC_WEATHERS)
    ax.set_ylabel("mAP50")
    ax.set_ylim(0, 1)
    ax.set_title("Per-weather mAP50 by experiment")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_per_weather.png", dpi=150)
    plt.close(fig)


def plot_class_weather_heatmap(summary: dict, out_dir: Path) -> None:
    """Write a per-class x per-weather mAP50 heatmap for each experiment."""
    ensure_dir(out_dir)
    for base, entry in sorted(summary.items()):
        if "fold_agg" in entry:
            per_weather = entry["fold_agg"].get("per_weather", {})
            getter = lambda b, c: b.get("per_class", {}).get(c, {}).get("mAP50", {}).get("mean")
        elif "official" in entry:
            per_weather = entry["official"].get("per_weather", {})
            getter = lambda b, c: b.get("per_class", {}).get(c, {}).get("mAP50")
        elif "single" in entry:
            per_weather = entry["single"].get("per_weather", {})
            getter = lambda b, c: b.get("per_class", {}).get(c, {}).get("mAP50")
        else:
            continue

        weathers = [w for w in ACDC_WEATHERS if w in per_weather]
        classes = [
            c for c in CLASS_ORDER if any(c in per_weather[w].get("per_class", {}) for w in weathers)
        ]
        if not weathers or not classes:
            continue
        data = [[getter(per_weather[w], c) or 0.0 for w in weathers] for c in classes]

        fig, ax = plt.subplots(figsize=(1.6 * len(weathers) + 2.5, 0.6 * len(classes) + 2))
        image = ax.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(weathers)))
        ax.set_xticklabels(weathers)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(weathers)):
                ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        ax.set_title(f"Per-class mAP50 by weather — {base}")
        fig.colorbar(image, ax=ax, label="mAP50")
        fig.tight_layout()
        fig.savefig(out_dir / f"class_weather_{base}.png", dpi=150)
        plt.close(fig)


def _plot_one_experiment(exp_dir: Path) -> None:
    """Generate figures for a single experiment directory."""
    figures = exp_dir / "figures"
    train_csv = exp_dir / "train" / "results.csv"
    if train_csv.exists():
        plot_training_curves(train_csv, figures / "training_curves.png")

    eval_dir = exp_dir / "eval"
    reports = sorted(eval_dir.glob("*.json")) if eval_dir.exists() else []
    official = [p for p in reports if p.stem.endswith("_official")]
    chosen = official[0] if official else (reports[0] if reports else None)
    if chosen is not None:
        report = json.loads(chosen.read_text(encoding="utf-8"))
        plot_experiment_bars(report, figures / f"bars_{chosen.stem}.png")


def main() -> None:
    """Generate per-experiment and cross-experiment figures."""
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("--exp", default=None, help="Only this experiment ID.")
    args = parser.parse_args()

    if args.exp:
        _plot_one_experiment(EXPERIMENTS_DIR / args.exp)
        print(f"[visualize] figures for {args.exp}")
        return

    if EXPERIMENTS_DIR.exists():
        for exp_dir in sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()):
            _plot_one_experiment(exp_dir)
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        plot_comparison(summary, SUMMARY_DIR / "figures")
        plot_class_weather_heatmap(summary, SUMMARY_DIR / "figures")
        print(f"[visualize] comparison figures -> {SUMMARY_DIR / 'figures'}")
    else:
        print("[visualize] no summary.json yet; run src/aggregate.py first")


if __name__ == "__main__":
    main()
