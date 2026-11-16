"""Thin, provenance-aware adapter around the unchanged official possession scorer.

This module does not implement an alternative spellF1. Detailed reports may contain
match-derived information and should be stored in the local private output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "vendor" / "mtah"
OFFICIAL_REVISION = "3de6b60aeabd2475d95879a651d38b72bc4fba00"
OFFICIAL_SCORER_SHA256 = "341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d"
PUBLIC_GT = OFFICIAL_ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
REFERENCE_KINDS = ("official_public", "independent_human", "pseudo_label", "synthetic")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def official_metrics():
    path = OFFICIAL_ROOT / "evaluation/possession_metrics.py"
    if sha256(path) != OFFICIAL_SCORER_SHA256:
        raise RuntimeError("Official scorer content changed; review and explicitly repin before evaluation")
    spec = importlib.util.spec_from_file_location("second_zero_official_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Official scorer is unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_submission(path: str | Path, n_expected: int = 199319) -> dict[str, Any]:
    """Run official validation, then enforce the exact canonical export contract."""
    path = Path(path)
    pm = official_metrics()
    table = pm.read_label_csv(str(path))
    reason = pm.validate_submission(table, n_expected)
    if reason:
        raise ValueError(f"Invalid submission: {reason}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        if handle.readline().rstrip("\r\n") != "frame,label":
            raise ValueError("Canonical export requires exactly the header frame,label")
    if list(table.columns) != ["frame", "label"]:
        raise ValueError("Canonical export permits only frame,label columns")
    if not table["label"].isin(pm.LABELS).all():
        raise ValueError("Canonical export requires uppercase labels without whitespace")
    return {"valid": True, "rows": len(table), "sha256": sha256(path),
            "validator": "official possession_metrics.validate_submission + canonical contract"}


def evaluate_submission(
    prediction_path: str | Path,
    ground_truth: str | Path = "public",
    *,
    reference_kind: str | None = None,
    bootstrap_n: int = 0,
) -> dict[str, Any]:
    """Evaluate using the official implementation and label the evidence correctly.

    Custom reference files require an explicit provenance kind so that model
    agreement can never be silently presented as independent ground-truth accuracy.
    Neither this module nor inference creates reference annotations.
    """
    if bootstrap_n < 0:
        raise ValueError("bootstrap_n must be nonnegative")
    is_public = str(ground_truth) == "public"
    gt_path = PUBLIC_GT if is_public else Path(ground_truth).resolve()
    if reference_kind is None:
        if not is_public:
            raise ValueError("Custom references require an explicit reference_kind")
        reference_kind = "official_public"
    if reference_kind not in REFERENCE_KINDS:
        raise ValueError(f"Unknown reference_kind: {reference_kind}")
    if reference_kind == "official_public" and sha256(gt_path) != sha256(PUBLIC_GT):
        raise ValueError("official_public must be the unchanged organizer public reference")
    pm = official_metrics()
    gt_table = pm.read_label_csv(str(gt_path))
    reason = pm.validate_submission(gt_table, len(gt_table))
    if reason:
        raise ValueError(f"Invalid reference: {reason}")
    validation = validate_submission(prediction_path, len(gt_table))
    pred_table = pm.read_label_csv(str(prediction_path))
    gt_codes, pred_codes = pm.label_codes(gt_table), pm.label_codes(pred_table)
    metrics = pm.evaluate(gt_codes, pred_codes, bootstrap_n=bootstrap_n)
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    raw_segments = pm.segments(pred_codes, runs)
    scored_segments = pm.segments(filled, runs)
    interpretation = {
        "official_public": "Public semantic/sanity evaluation only; not a full-match score or proof of superiority.",
        "independent_human": "Accuracy against the declared independent human reference; independence must be established by its annotation provenance.",
        "pseudo_label": "Agreement with model-generated pseudo-labels; not an independent accuracy measurement.",
        "synthetic": "Synthetic behavior test; not measured match accuracy.",
    }[reference_kind]
    return {
        "schema_version": 1,
        "reference_kind": reference_kind,
        "interpretation": interpretation,
        "prediction": {"path": str(Path(prediction_path).resolve()), **validation},
        "reference": {"path": str(gt_path.resolve()), "sha256": sha256(gt_path),
                      "graded_runs_half_open": runs},
        "official_scorer": {
            "repository": "https://github.com/metrica-sports/more-than-a-hack-2027",
            "revision": OFFICIAL_REVISION,
            "sha256": sha256(OFFICIAL_ROOT / "evaluation/possession_metrics.py"),
            "spell_iou_thresholds": list(pm.SPELL_IOUS),
        },
        "metrics": metrics,
        "abstention_diagnostics": {
            "raw_segments_in_graded_runs": len(raw_segments),
            "scored_segments_after_official_fill": len(scored_segments),
            "filled_frame_count": int(np.count_nonzero(filled != pred_codes)),
            "policy": "Raw X is a frame-level miss; temporal metrics forward-fill X and back-fill run heads, independently inside each graded run.",
        },
    }


def compare_public_baselines() -> dict[str, Any]:
    """Score the four public organizer predictions, without reading match assets."""
    reports = {}
    for name in ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125"):
        reports[name] = evaluate_submission(OFFICIAL_ROOT / f"evaluation/results/{name}.csv")
    return {"schema_version": 1, "reference_kind": "official_public",
            "interpretation": "These are local evaluations of organizer-provided predictions, not regeneration from tracking.",
            "reports": reports}


def write_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prediction", nargs="?")
    parser.add_argument("--gt", default="public")
    parser.add_argument("--reference-kind", choices=REFERENCE_KINDS)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public-baselines", action="store_true")
    parser.add_argument("--quiet", action="store_true", help="Print only completion, without derived scores")
    args = parser.parse_args()
    if args.public_baselines:
        report = compare_public_baselines()
    elif args.prediction:
        report = evaluate_submission(args.prediction, args.gt, reference_kind=args.reference_kind,
                                     bootstrap_n=args.bootstrap)
    else:
        parser.error("provide a prediction CSV or --public-baselines")
    write_report(report, args.out)
    if not args.quiet and "metrics" in report:
        metrics = report["metrics"]
        print(f"spellF1={metrics['spell_f1_mean']:.6f} macroF1={metrics['macro_f1']:.6f}")
    print("Evaluation complete; report saved locally.")


if __name__ == "__main__":
    main()
