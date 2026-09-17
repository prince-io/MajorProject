"""Aggregate per-fold evaluation reports into mean +/- std summaries and tables.

Reads ``results/experiments/*/eval/*.json`` written by ``src/eval.py``, groups
reports by experiment (stripping ``_official`` / ``_foldN`` suffixes), and computes
mean and standard deviation across the 5 CV folds for overall, per-weather, and
per-class metrics. Writes:

- ``results/summary/summary.json``   - full nested aggregation
- ``results/summary/per_class.csv``  - flat per-class table (wide format)
- ``results/summary/per_class.md``   - per-class x per-weather Markdown tables

Usage::

    python src/aggregate.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EXPERIMENTS_DIR, RESULTS_DIR, ensure_dir  # noqa: E402

METRICS_DIR = RESULTS_DIR / "metrics"  # legacy flat location (kept for compatibility)
SUMMARY_PATH = RESULTS_DIR / "summary" / "summary.json"
PER_CLASS_CSV = RESULTS_DIR / "summary" / "per_class.csv"
PER_CLASS_MD = RESULTS_DIR / "summary" / "per_class.md"

METRIC_KEYS: tuple[str, ...] = ("mAP50", "mAP50-95", "precision", "recall")
CLASS_ORDER: tuple[str, ...] = ("person", "rider", "car", "truck", "bus", "bicycle")
FOLD_RE = re.compile(r"_fold\d+$")
SUFFIX_RE = re.compile(r"_(official|fold\d+)$")


def _agg(values: list[float | None]) -> dict:
    """Mean/std (sample) over the non-None values."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def _agg_block(blocks: list[dict]) -> dict:
    """Aggregate a list of metric blocks (overall or per-weather)."""
    out: dict = {"metrics": {}, "per_class": {}}
    for metric in METRIC_KEYS:
        out["metrics"][metric] = _agg([b.get(metric) for b in blocks])
    classes: set[str] = set()
    for block in blocks:
        classes |= set(block.get("per_class", {}).keys())
    for name in sorted(classes):
        out["per_class"][name] = {
            metric: _agg([b.get("per_class", {}).get(name, {}).get(metric) for b in blocks])
            for metric in METRIC_KEYS
        }
    return out


def _agg_per_weather(reports: list[dict]) -> dict:
    """Aggregate the per_weather section across reports."""
    weathers: set[str] = set()
    for report in reports:
        weathers |= set(report.get("per_weather", {}).keys())
    return {
        weather: _agg_block(
            [r["per_weather"][weather] for r in reports if weather in r.get("per_weather", {})]
        )
        for weather in sorted(weathers)
    }


def _fmt(block: dict, metric: str = "mAP50") -> str:
    """Format a mean/std pair (folds) or a raw value (official/single)."""
    stat = block.get("metrics", {}).get(metric)
    if stat is None:
        return "—"
    if isinstance(stat, dict):
        if stat.get("mean") is None:
            return "—"
        return f"{stat['mean']:.4f} ± {stat['std']:.4f}"
    return f"{stat:.4f}"


def _class_stat(block: dict, cls: str, metric: str) -> dict:
    """Return the {mean,std,n} for one class metric from an aggregated or raw block."""
    value = block.get("per_class", {}).get(cls, {}).get(metric)
    if isinstance(value, dict):
        return value
    return {"mean": value, "std": None, "n": 1 if value is not None else 0}


def _iter_eval_blocks(entry: dict):
    """Yield (eval_name, weather, block) for every eval section of a summary entry."""
    for eval_name, key in (("5fold", "fold_agg"), ("official", "official"), ("single", "single")):
        if key not in entry:
            continue
        section = entry[key]
        yield eval_name, "overall", section["overall"]
        for weather, block in section.get("per_weather", {}).items():
            yield eval_name, weather, block


def _ordered_classes(block: dict) -> list[str]:
    """Class names present in a block, in the canonical unified order."""
    present = set(block.get("per_class", {}).keys())
    return [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))


def _cell(block: dict, cls: str, metric: str = "mAP50") -> str:
    """Format one class metric as ``mean`` or ``mean±std``."""
    stat = _class_stat(block, cls, metric)
    if stat.get("mean") is None:
        return "—"
    if stat.get("std") is None:
        return f"{stat['mean']:.3f}"
    return f"{stat['mean']:.3f}±{stat['std']:.3f}"


def write_tables(summary: dict) -> None:
    """Write the flat CSV and the per-class x per-weather Markdown tables."""
    rows: list[dict] = []
    for base, entry in sorted(summary.items()):
        for eval_name, weather, block in _iter_eval_blocks(entry):
            for cls in _ordered_classes(block):
                row: dict = {"experiment": base, "eval": eval_name, "weather": weather, "class": cls}
                for metric in METRIC_KEYS:
                    stat = _class_stat(block, cls, metric)
                    row[metric] = stat.get("mean")
                    row[f"{metric}_std"] = stat.get("std")
                    row[f"{metric}_n"] = stat.get("n")
                rows.append(row)

    fieldnames = ["experiment", "eval", "weather", "class"] + [
        f"{metric}{suffix}" for metric in METRIC_KEYS for suffix in ("", "_std", "_n")
    ]
    ensure_dir(PER_CLASS_CSV.parent)
    with PER_CLASS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: (round(value, 4) if isinstance(value, float) else value) for key, value in row.items()}
            )

    lines = ["# Per-class evaluation", ""]
    for base, entry in sorted(summary.items()):
        lines.append(f"## {base}")
        lines.append("")
        for eval_name, key in (("5-fold", "fold_agg"), ("official", "official"), ("single", "single")):
            if key not in entry:
                continue
            section = entry[key]
            weathers = ["overall"] + sorted(section.get("per_weather", {}).keys())
            blocks = {"overall": section["overall"], **section.get("per_weather", {})}
            lines.append(f"### {eval_name}")
            lines.append("")
            lines.append("| class | " + " | ".join(weathers) + " |")
            lines.append("|" + "---|" * (len(weathers) + 1))
            for cls in _ordered_classes(blocks["overall"]):
                lines.append(
                    f"| {cls} | " + " | ".join(_cell(blocks[w], cls) for w in weathers) + " |"
                )
            lines.append("")
    PER_CLASS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Read reports, aggregate folds, write and print the summary."""
    reports: list[dict] = []
    seen: set[Path] = set()
    candidates = sorted(EXPERIMENTS_DIR.glob("*/eval/*.json")) + sorted(METRICS_DIR.glob("*.json"))
    for path in candidates:
        if path.name == "summary.json" or path in seen:
            continue
        seen.add(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "overall" in data and "run" in data:
            reports.append(data)

    groups: dict[str, list[dict]] = defaultdict(list)
    for report in reports:
        groups[SUFFIX_RE.sub("", report["run"])].append(report)

    summary: dict = {}
    for base, members in sorted(groups.items()):
        folds = [r for r in members if FOLD_RE.search(r["run"])]
        official = [r for r in members if r["run"].endswith("_official")]
        singles = [r for r in members if r not in folds and r not in official]

        entry: dict = {"folds": len(folds)}
        if folds:
            entry["fold_agg"] = {
                "overall": _agg_block([r["overall"] for r in folds]),
                "per_weather": _agg_per_weather([{"per_weather": r.get("per_weather", {})} for r in folds]),
            }
        if official:
            entry["official"] = {
                "overall": official[0]["overall"],
                "per_weather": official[0].get("per_weather", {}),
            }
        if singles:
            entry["single"] = {
                "overall": singles[0]["overall"],
                "per_weather": singles[0].get("per_weather", {}),
            }
        summary[base] = entry

    ensure_dir(SUMMARY_PATH.parent)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tables(summary)

    print(f"{'experiment':<22} {'eval':<10} {'mAP50':<18} {'mAP50-95':<18}")
    print("-" * 70)
    for base, entry in summary.items():
        if "fold_agg" in entry:
            overall = entry["fold_agg"]["overall"]
            print(
                f"{base:<22} {'5-fold':<10} {_fmt(overall):<18} "
                f"{_fmt(overall, 'mAP50-95'):<18}"
            )
        if "official" in entry:
            overall = {"metrics": entry["official"]["overall"]}
            print(
                f"{base:<22} {'official':<10} {_fmt(overall):<18} "
                f"{_fmt(overall, 'mAP50-95'):<18}"
            )
        if "single" in entry:
            overall = {"metrics": entry["single"]["overall"]}
            print(
                f"{base:<22} {'single':<10} {_fmt(overall):<18} "
                f"{_fmt(overall, 'mAP50-95'):<18}"
            )
    print(f"\n[aggregate] summary -> {SUMMARY_PATH}")
    print(f"[aggregate] tables  -> {PER_CLASS_CSV} , {PER_CLASS_MD}")


if __name__ == "__main__":
    main()
