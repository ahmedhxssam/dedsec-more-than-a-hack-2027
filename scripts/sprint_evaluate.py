"""Independent official-score audit for frozen two-hour-sprint predictions.

Only public development or explicitly listed non-reserved external matches are
accepted. This evaluator neither fits nor changes predictions and never promotes
a candidate. Whole prediction files and this source are pinned before labels open.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "private/improvement/runtime"), str(ROOT)]
sys.dont_write_bytecode = True

import numpy as np

from second_zero.evaluate import PUBLIC_GT, official_metrics, sha256, validate_submission

ALLOWED_MATCHES = {"J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ"}
HALVES = ("firstHalf", "secondHalf")
CONDITIONS = ("clean", "corrupted")
METRICS = ("spell_f1_mean", "macro_f1")
REFERENCE_READS: set[str] = set()


def now():
    return datetime.now(timezone.utc).isoformat()


def guard(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0])
    if "J03WR9" in path:
        raise PermissionError("Reserved final match is prohibited in this sprint evaluator")
    if path.endswith(".truth.npy") or path == str(PUBLIC_GT):
        REFERENCE_READS.add(path)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def decode_candidate(path, array):
    if path.suffix == ".csv":
        validation = validate_submission(path)
        values = official_metrics().label_codes(official_metrics().read_label_csv(path))
    else:
        with np.load(path, allow_pickle=False) as z:
            values = z[array].copy()
        validation = {"rows": len(values), "sha256": sha256(path)}
    if values.ndim != 1 or not np.isin(values, (0, 1, 2, 3)).all():
        raise ValueError("Invalid canonical H/A/D/X prediction: " + str(path))
    return values.astype(np.int8), validation


def event_audit(pm, truth, predicted):
    runs = pm.valid_runs(truth)
    refs = pm.possession_spells(truth, runs)
    preds = pm.possession_spells(pm.fill_abstentions(predicted, runs), runs)
    overlaps = pm.spell_overlaps(refs, preds)
    flags = {str(t): pm.matched_gt_flags(overlaps, len(refs), t) for t in pm.SPELL_IOUS}
    events = []
    for i, (start, stop, team) in enumerate(refs):
        best = max(((float(iou), int(pi)) for iou, gi, pi in overlaps if gi == i), default=(0., None))
        values = predicted[start:stop]
        events.append({"start": int(start), "stop": int(stop), "label": pm.LABELS[team],
                       "frames": int(stop-start), "correct_frames": int((values == team).sum()),
                       "dead_frames": int((values == 2).sum()), "wrong_team_frames": int((values == 1-team).sum()),
                       "best_same_team_iou": best[0],
                       "best_prediction": None if best[1] is None else [int(x) for x in preds[best[1]]],
                       "matched": {t: bool(f[i]) for t, f in flags.items()}})
    short = np.array([b-a <= 125 for a, b, _ in refs], bool)
    return {"reference_spells": len(refs), "predicted_spells": len(preds),
            "short_reference_spells": int(short.sum()),
            "short_matched_iou05": int((flags["0.5"] & short).sum()),
            "matched_per_iou": {t: int(f.sum()) for t, f in flags.items()}, "events": events}


def score(pm, truth, predicted, include_events=False):
    if predicted.shape != truth.shape:
        raise ValueError(f"Prediction shape {predicted.shape} differs from truth {truth.shape}")
    valid = truth < 3
    result = {"metrics": pm.evaluate(truth, predicted, bootstrap_n=0),
              "graded_frames": int(valid.sum()),
              "false_D_frames": int(((predicted == 2) & (truth < 2)).sum()),
              "missed_D_frames": int(((predicted != 2) & (truth == 2)).sum()),
              "wrong_team_frames": int(((truth < 2) & (predicted < 2) & (truth != predicted)).sum()),
              "correct_frames": int((valid & (truth == predicted)).sum())}
    if include_events:
        result["events"] = event_audit(pm, truth, predicted)
    return result


def public_audit(args, candidates, out):
    pm = official_metrics()
    truth = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    reports = {}
    for name, path in candidates.items():
        predicted, validation = decode_candidate(path, args.array)
        row = score(pm, truth, predicted, include_events=True)
        row.update(prediction_path=str(path), validation=validation)
        reports[name] = row
    result = {"reference_kind": "official_public_development", "reference_sha256": sha256(PUBLIC_GT),
              "interpretation": "Repeatedly used 7,500-frame target development mask; not full-match or independent accuracy.",
              "reports": reports}
    return result


def external_audit(args, candidates, out):
    pm = official_metrics()
    if not set(args.matches) <= ALLOWED_MATCHES:
        raise ValueError("Only explicitly permitted development matches may be scored")
    truth_cache = {}
    truth_hashes = {}
    for match in args.matches:
        for half in HALVES:
            path = ROOT / "private/improvement/external/prepared" / f"{match}-{half}.truth.npy"
            truth_cache[(match, half)] = np.load(path, allow_pickle=False)
            truth_hashes[str(path)] = sha256(path)
    reports = {}
    for name, folder in candidates.items():
        views = {}
        for match in args.matches:
            for half in HALVES:
                for condition in CONDITIONS:
                    key = f"{match}-{half}-{condition}"
                    path = folder / f"{key}.npz"
                    predicted, validation = decode_candidate(path, args.array)
                    row = score(pm, truth_cache[(match, half)], predicted, args.events)
                    row.update(match=match, half=half, condition=condition, validation=validation)
                    views[key] = row
        match_means = {match: {metric: float(np.mean([row["metrics"][metric] for row in views.values() if row["match"] == match]))
                              for metric in METRICS} for match in args.matches}
        condition_means = {condition: {metric: float(np.mean([row["metrics"][metric] for row in views.values() if row["condition"] == condition]))
                                      for metric in METRICS} for condition in CONDITIONS}
        overall = {metric: float(np.mean([row[metric] for row in match_means.values()])) for metric in METRICS}
        reports[name] = {"views": views, "match_means": match_means, "condition_means": condition_means, "overall": overall}
    return {"reference_kind": "external_development", "truth_sha256": truth_hashes,
            "interpretation": "Equal means over halves/paired clean-corrupted conditions within each match, then equal matches; paired conditions are not independent matches. Prior development exposure persists.",
            "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("public", "external"))
    parser.add_argument("--candidate", action="append", required=True, help="NAME=CSV/NPZ (public) or NAME=DIRECTORY (external)")
    parser.add_argument("--array", default="predicted")
    parser.add_argument("--matches", nargs="+", default=["J03WQQ"])
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.addaudithook(guard)
    candidates = {}
    for spec in args.candidate:
        name, path = spec.split("=", 1)
        if name in candidates:
            raise ValueError("Duplicate candidate name")
        candidates[name] = Path(path).resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / "second_zero/evaluate.py", ROOT / "vendor/mtah/evaluation/possession_metrics.py"]
    if args.mode == "public":
        paths += list(candidates.values())
    else:
        if not set(args.matches) <= ALLOWED_MATCHES:
            raise ValueError("Unpermitted reference match")
        paths += [folder / f"{match}-{half}-{condition}.npz" for folder in candidates.values()
                  for match in args.matches for half in HALVES for condition in CONDITIONS]
    pinned = {str(path): sha256(path) for path in paths}
    freeze = {"created_utc": now(), "mode": args.mode, "array": args.array,
              "candidates": {k: str(v) for k, v in candidates.items()}, "input_sha256": pinned,
              "reserved_final_access_prohibited": True, "reference_reads_before_freeze": sorted(REFERENCE_READS)}
    if REFERENCE_READS:
        raise RuntimeError("Labels read before prediction freeze")
    write_new(out / "pre_score_freeze.json", freeze)
    result = public_audit(args, candidates, out) if args.mode == "public" else external_audit(args, candidates, out)
    for path, digest in pinned.items():
        if sha256(path) != digest:
            raise RuntimeError("Frozen prediction/source changed during scoring: " + path)
    result.update(created_utc=now(), freeze_sha256=sha256(out / "pre_score_freeze.json"),
                  reference_reads=sorted(REFERENCE_READS), reserved_final_accessed=False,
                  input_hashes_unchanged=True)
    write_new(out / "results.json", result)
    for name, row in result["reports"].items():
        metrics = row["metrics"] if args.mode == "public" else row["overall"]
        print(name, " ".join(f"{metric}={metrics[metric]:.9f}" for metric in METRICS), flush=True)


if __name__ == "__main__":
    main()
