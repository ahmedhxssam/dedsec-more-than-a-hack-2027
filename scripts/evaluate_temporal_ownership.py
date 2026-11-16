"""Compare frozen temporal ownership evidence with exactly fixed live/dead masks.

All CSVs written here are full-length experimental outputs, never submissions.
No training, calibration, threshold selection, or source-video extraction occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "private/improvement/runtime"), str(ROOT)]

import numpy as np

from second_zero.common import json_write, sha256
from second_zero.contextual import binary_decode
from second_zero.focused_boundaries import touch_decode
from second_zero.evaluate import (OFFICIAL_ROOT, PUBLIC_GT, evaluate_submission,
                                  official_metrics, validate_submission)
from scripts.evaluate_focused_pass import best_gain, diagnostic

TEAM_COST = 6.0
BASELINES = {
    "retained": (ROOT / "DedSec.csv", ROOT / "runs/focused-v1/retained/evidence.npz"),
    "combined": (ROOT / "runs/focused-v1/combined/predictions.csv",
                 ROOT / "runs/focused-v1/combined/evidence.npz"),
}


def sigmoid(logit):
    x = np.asarray(logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def fixed_state_owner(baseline, logit, *, framewise=False, touch_support=None):
    """Replace H/A only, resetting the unchanged decoder at every D/X island."""
    baseline, logit = np.asarray(baseline), np.asarray(logit)
    if baseline.ndim != 1 or logit.shape != baseline.shape:
        raise ValueError("Baseline and logit must be matching one-dimensional arrays")
    if not np.isin(baseline, [0, 1, 2, 3]).all() or not np.isfinite(logit).all():
        raise ValueError("Invalid labels or nonfinite ownership evidence")
    if touch_support is not None:
        touch_support = np.asarray(touch_support)
        if touch_support.shape != (len(baseline), 2) or not np.isfinite(touch_support).all():
            raise ValueError("Frozen touch support must be a finite n-by-2 matrix")
    out = baseline.copy()
    live = (baseline == 0) | (baseline == 1)
    edges = np.r_[0, np.flatnonzero(live[1:] != live[:-1]) + 1, len(live)]
    for start, stop in zip(edges[:-1], edges[1:]):
        if live[start]:
            if framewise:
                out[start:stop] = (logit[start:stop] > 0).astype(np.int8)
            elif touch_support is None:
                out[start:stop] = binary_decode(logit[start:stop], TEAM_COST)
            else:
                out[start:stop] = touch_decode(logit[start:stop], touch_support[start:stop], TEAM_COST)
    if not np.array_equal(out[~live], baseline[~live]):
        raise AssertionError("Ownership experiment changed a fixed D/X label")
    return out


def evidence_metrics(gt, probability_a, mask):
    """Binary probabilities are assessed only on reference H/A frames."""
    eligible = np.asarray(mask, dtype=bool) & np.isin(gt, [0, 1])
    truth = np.asarray(gt)[eligible]
    p = np.clip(np.asarray(probability_a)[eligible], 1e-12, 1 - 1e-12)
    if not len(p):
        return {"frames": 0, "nll": None, "brier": None, "accuracy": None}
    return {"frames": int(len(p)), "home_frames": int((truth == 0).sum()),
            "away_frames": int((truth == 1).sum()),
            "nll": float(-np.mean(np.where(truth == 1, np.log(p), np.log1p(-p)))),
            "brier": float(np.mean((p - truth) ** 2)),
            "accuracy": float(np.mean((p > .5) == truth))}


def frame_changes(gt, base, pred, mask):
    changed = (pred != base) & mask
    corrected = changed & (pred == gt) & (base != gt)
    damaged = changed & (pred != gt) & (base == gt)
    still_wrong = changed & (pred != gt) & (base != gt)
    if not np.array_equal(changed, corrected | damaged | still_wrong):
        raise AssertionError("Frame change categories do not partition changes")
    return {"changed_frames": int(changed.sum()), "corrected_frames": int(corrected.sum()),
            "newly_damaged_frames": int(damaged.sum()),
            "still_wrong_changed_frames": int(still_wrong.sum()),
            "net_correct_frames": int(corrected.sum() - damaged.sum())}


def possession_effects(pm, gt, base, pred):
    runs = pm.valid_runs(gt)
    gs = pm.possession_spells(gt, runs)
    base = pm.fill_abstentions(base, runs)
    pred = pm.fill_abstentions(pred, runs)
    bm = pm.matched_gt_flags(pm.spell_overlaps(gs, pm.possession_spells(base, runs)), len(gs), .5)
    nm = pm.matched_gt_flags(pm.spell_overlaps(gs, pm.possession_spells(pred, runs)), len(gs), .5)
    rows = []
    for i, (start, stop, team) in enumerate(gs):
        bcorrect, ncorrect = base[start:stop] == team, pred[start:stop] == team
        rows.append({"id": i + 1, "reference": [int(start), int(stop), int(team)],
                     "short_le_125f": bool(stop - start <= 125),
                     "baseline_matched_iou50": bool(bm[i]), "new_matched_iou50": bool(nm[i]),
                     "match_effect": ("preserved" if bm[i] else "recovered") if nm[i]
                                     else ("lost" if bm[i] else "still_unmatched"),
                     "baseline_correct_frames": int(bcorrect.sum()),
                     "new_correct_frames": int(ncorrect.sum()),
                     "corrected_frames": int((~bcorrect & ncorrect).sum()),
                     "newly_damaged_frames": int((bcorrect & ~ncorrect).sum()),
                     "previously_all_frames_correct": bool(bcorrect.all())})
    return {"matched_preserved": int((bm & nm).sum()), "matched_lost": int((bm & ~nm).sum()),
            "matched_recovered": int((~bm & nm).sum()),
            "previously_all_correct_spells_damaged": sum(r["previously_all_frames_correct"] and
                                                        r["newly_damaged_frames"] > 0 for r in rows),
            "spells": rows}


def fixed_boundary_changes(pm, gt, base, pred):
    """D boundaries can retain timing but change their scored H/A label pair."""
    runs = pm.valid_runs(gt)
    base, pred = pm.fill_abstentions(base, runs), pm.fill_abstentions(pred, runs)
    old, new = pm.transitions(base, runs), pm.transitions(pred, runs)
    result = {}
    for name, predicate in [("onset", lambda t: t[1] in [pm.H, pm.A] and t[2] == pm.D),
                            ("restart", lambda t: t[1] == pm.D and t[2] in [pm.H, pm.A])]:
        before = {int(t[0]): [int(t[1]), int(t[2])] for t in old if predicate(t)}
        after = {int(t[0]): [int(t[1]), int(t[2])] for t in new if predicate(t)}
        result[name] = {"timings_unchanged": sorted(before) == sorted(after),
                        "changed_label_pairs": [{"frame": frame, "before": before[frame],
                                                 "after": after[frame]}
                                                for frame in sorted(before.keys() & after.keys())
                                                if before[frame] != after[frame]]}
        if not result[name]["timings_unchanged"]:
            raise AssertionError("Fixed D mask changed live/dead boundary timing")
    return result


def short_evidence(pm, gt, logit, base, *, destination_support=False):
    rows = []
    for start, stop, team in pm.possession_spells(gt, pm.valid_runs(gt)):
        if stop - start > 125:
            continue
        signed = np.asarray(logit[start:stop]) * (1 if team == pm.A else -1)
        live = np.isin(base[start:stop], [pm.H, pm.A])
        edges = np.r_[0, np.flatnonzero(live[1:] != live[:-1]) + 1, len(live)]
        live_gain = max((best_gain(signed[a:b]) for a, b in zip(edges[:-1], edges[1:]) if live[a]),
                        default=0.)
        rows.append({"reference": [int(start), int(stop), int(team)],
                     "duration_frames": int(stop - start), "fixed_live_frames": int(live.sum()),
                     "favorable_frames": int((signed > 0).sum()),
                     "signed_logit_sum": float(signed.sum()),
                     "maximum_favorable_subsequence_gain": float(best_gain(signed)),
                     "signed_logit_sum_on_fixed_live": float(signed[live].sum()),
                     "maximum_favorable_subsequence_gain_within_fixed_live_island": float(live_gain),
                     "exceeds_12_logit_gain_within_reference_live_interval": bool(live_gain > 2 * TEAM_COST),
                     "can_exceed_exact_two_switch_cost": None if destination_support else bool(live_gain > 2 * TEAM_COST),
                     "two_switch_cost": None if destination_support else 2 * TEAM_COST,
                     "two_switch_cost_range": [TEAM_COST, 2 * TEAM_COST] if destination_support else [2 * TEAM_COST, 2 * TEAM_COST],
                     "gain_12_interpretation": "Conservative upper cost bound only; combined uses frozen destination-dependent costs3..6 per switch."
                                               if destination_support else "Exact two-switch cost for an interior retained live-island interval."})
    return rows


def write_csv(path, codes):
    labels = np.asarray(["H", "A", "D", "X"])[codes]
    with path.open("w") as stream:
        stream.write("frame,label\n")
        stream.writelines(f"{i},{label}\n" for i, label in enumerate(labels))
    return validate_submission(path, len(labels))


def load_probabilities(path, n):
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in ["owner", "raw_owner", "logit", "visual_logit", "regions"]}
    for key in ["owner", "raw_owner"]:
        p = result[key]
        if p.shape != (n, 2) or not np.isfinite(p).all() or not ((0 <= p) & (p <= 1)).all():
            raise ValueError(f"Invalid {key} probability matrix")
        if not np.allclose(p.sum(axis=1), 1., atol=1e-6):
            raise ValueError(f"{key} probabilities do not sum to one")
    for key in ["logit", "visual_logit"]:
        if result[key].shape != (n,) or not np.isfinite(result[key]).all():
            raise ValueError(f"Invalid {key}")
    if not np.allclose(sigmoid(result["logit"]), result["owner"][:, 1], atol=1e-6):
        raise ValueError("Calibrated owner probability and decoder logit disagree")
    regions = result["regions"]
    if regions.ndim != 2 or regions.shape[1] != 2:
        raise ValueError("Regions must contain half-open source-frame intervals")
    coverage = np.zeros(n, bool)
    previous_stop = 0
    for start, stop in regions:
        if int(start) != start or int(stop) != stop or not (previous_stop <= start < stop <= n):
            raise ValueError("Inference regions are not ordered nonoverlapping integer ranges")
        coverage[int(start):int(stop)] = True
        previous_stop = stop
    result["coverage"] = coverage
    return result


def external_event_receipt(run, metadata, pm):
    """Add event accounting to existing external results, without fitting anything."""
    plan = metadata["protocol.json"]["contents"]
    existing = metadata["external_results.json"]["contents"]["halves"]
    prepared = ROOT / "private/improvement/external/prepared"
    halves, grouped = {}, {}
    expected = [f"{match}-{half}-{variant}" for match in plan["split"]["development"]
                for half in ["firstHalf", "secondHalf"] for variant in ["clean", "corrupted"]]
    if set(existing) != set(expected):
        raise ValueError("External receipt expects precisely the established development halves")
    for key in expected:
        match, half, variant = key.split("-")
        truth_path = prepared / f"{match}-{half}.truth.npy"
        gt = np.load(truth_path, allow_pickle=False)
        prediction_path = run / (key + ".npz")
        with np.load(prediction_path, allow_pickle=False) as saved:
            base, pred, logit = saved["incumbent"], saved["predicted"], saved["logit"]
        if gt.shape != base.shape or base.shape != pred.shape or pred.shape != logit.shape:
            raise ValueError("External truth/prediction frame alignment mismatch")
        if not np.array_equal(base[base >= pm.D], pred[base >= pm.D]):
            raise ValueError("External temporal model altered a fixed D/X label")
        if not np.array_equal(fixed_state_owner(base, logit), pred):
            raise ValueError("External saved decoded decisions fail exact evidence replay")
        argmax = fixed_state_owner(base, logit, framewise=True)
        effect = possession_effects(pm, gt, base, pred)
        shorts = [row for row in effect["spells"] if row["short_le_125f"]]
        short = {"support": len(shorts),
                 "incumbent_matched": sum(row["baseline_matched_iou50"] for row in shorts),
                 "temporal_matched": sum(row["new_matched_iou50"] for row in shorts),
                 "preserved": sum(row["match_effect"] == "preserved" for row in shorts),
                 "lost": sum(row["match_effect"] == "lost" for row in shorts),
                 "recovered": sum(row["match_effect"] == "recovered" for row in shorts),
                 "corrected_frames": sum(row["corrected_frames"] for row in shorts),
                 "newly_damaged_frames": sum(row["newly_damaged_frames"] for row in shorts)}
        role = existing[key]["role"]
        item = {"role": role, "variant": variant,
                "truth_sha256": sha256(truth_path), "prediction_sha256": sha256(prediction_path),
                "short_spells": short, "frame_changes": frame_changes(gt, base, pred, gt != pm.X),
                "possession_effects": effect,
                "decoder_versus_argmax": frame_changes(gt, argmax, pred, gt != pm.X),
                "D_frame_changes": int(((base == pm.D) != (pred == pm.D)).sum()),
                "X_frame_changes": int(((base == pm.X) != (pred == pm.X)).sum())}
        halves[key] = item
        group = grouped.setdefault(role + "-" + variant,
                                   {"short_spells": {k: 0 for k in short},
                                    "frame_changes": {k: 0 for k in item["frame_changes"]},
                                    "all_possessions": {k: 0 for k in ["matched_preserved", "matched_lost", "matched_recovered",
                                                                       "previously_all_correct_spells_damaged"]}})
        for name in group:
            source = effect if name == "all_possessions" else item[name]
            for field in group[name]:
                group[name][field] += source[field]
    return {"scope": "Existing external development split only; calibration match is in-sample for temperature. Clean and corrupted views share underlying labels and are not independent samples.",
            "short_definition": "<=125 source frames (5s at25fps), official same-team greedy one-to-one IoU>=0.5.",
            "aggregate": grouped, "halves": halves, "final_match_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Evaluation output must be a fresh directory")
    # Calibration is supplied by the external-development training run, never fit here.
    metadata = {}
    for name in ["protocol.json", "calibration.json", "external_results.json", "freeze.json", "target_provenance.json"]:
        path = args.run / name
        metadata[name] = {"sha256": sha256(path), "contents": json.loads(path.read_text())}
    freeze = metadata["freeze.json"]["contents"]
    calibration = metadata["calibration.json"]["contents"]
    target = metadata["target_provenance.json"]["contents"]
    if (freeze["protocol_sha256"] != metadata["protocol.json"]["sha256"] or
            freeze["calibration_sha256"] != metadata["calibration.json"]["sha256"] or
            target["freeze_sha256"] != metadata["freeze.json"]["sha256"] or
            target["probabilities_sha256"] != sha256(args.run / "target_probabilities.npz")):
        raise ValueError("External calibration/frozen model/target prediction provenance chain differs")
    if (calibration["public_labels_used"] or calibration["validation_match_used"] or
            target["public_labels_read"] or not np.isfinite(calibration["temperature"]) or
            calibration["temperature"] <= 0 or freeze["temperature"] != calibration["temperature"]):
        raise ValueError("New ownership evidence lacks the specified external-only temperature calibration")
    expected = json.loads((ROOT / "configs/control_history_pass.json").read_text())["preserved_csv_sha256"]
    before = {name: sha256(ROOT / name) for name in expected}
    if before != expected:
        raise ValueError("Existing candidate hashes differ from the preserved pass")
    pm = official_metrics()
    baseline_codes = {name: pm.label_codes(pm.read_label_csv(path)) for name, (path, _) in BASELINES.items()}
    n = len(baseline_codes["retained"])
    new = load_probabilities(args.run / "target_probabilities.npz", n)
    cached = {}
    frozen_touch = {"retained": None}
    for name, (_, path) in BASELINES.items():
        with np.load(path, allow_pickle=False) as old:
            cached[name] = {key: old[key] for key in ["interaction_logit", "p_owner", "p_dead",
                                                     "native_ball_support", "native_image_reliability"]}
            if not np.array_equal(pm.label_codes(old["label"]), baseline_codes[name]):
                raise ValueError("Cached evidence and preserved candidate labels differ: " + name)
        if np.any(np.isin(baseline_codes[name], [pm.H, pm.A]) & ~new["coverage"]):
            raise ValueError("New evidence does not cover every baseline live frame")
    touch_path = ROOT / "runs/focused-v1/combined/focused_evidence.npz"
    with np.load(touch_path, allow_pickle=False) as old:
        frozen_touch["combined"] = old["touch_support"]
    replay = {}
    for name, base in baseline_codes.items():
        old_decode = fixed_state_owner(base, cached[name]["interaction_logit"], touch_support=frozen_touch[name])
        replay[name] = {"changed_frames": int((old_decode != base).sum()), "exact": bool(np.array_equal(old_decode, base))}
        if not replay[name]["exact"]:
            raise ValueError("Old evidence through frozen ownership decoder fails exact candidate replay: " + name)
    gate = .75 + .25 * (.75 * np.clip(cached["retained"]["native_image_reliability"], 0, 1) +
                         .25 * np.clip(cached["retained"]["native_ball_support"], 0, 1))
    if not np.allclose(new["visual_logit"], new["logit"] * gate, atol=1e-6):
        raise ValueError("Visual ablation differs from the existing bounded native gate")

    args.out.mkdir(parents=True, exist_ok=False)
    predictions = {}
    # Export all frozen inference arms before opening the public reference.
    for base_name, base in baseline_codes.items():
        for variant, logit_key, framewise in [("decoded", "logit", False),
                                              ("argmax", "logit", True),
                                              ("native_gate", "visual_logit", False)]:
            name = f"{base_name}_{variant}"
            folder = args.out / name
            folder.mkdir()
            codes = fixed_state_owner(base, new[logit_key], framewise=framewise, touch_support=frozen_touch[base_name])
            path = folder / "experimental_predictions.csv"
            validation = write_csv(path, codes)
            json_write(folder / "validation.json", validation)
            predictions[name] = (base_name, codes, path, logit_key)
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    mask = gt != pm.X
    with np.load(ROOT / "runs/main/inputs.npz", allow_pickle=False) as raw:
        ball_seen = np.isfinite(raw["ball_xy"]).all(axis=1)
    results = {"reference_kind": "official_public_development", "scored_frames": int(mask.sum()),
               "reference_sha256": sha256(PUBLIC_GT),
               "public_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
               "official_scorer_sha256": sha256(OFFICIAL_ROOT / "evaluation/possession_metrics.py"),
               "candidate_hashes_before": before, "target_probabilities_sha256": sha256(args.run / "target_probabilities.npz"),
               "baseline_evidence_sha256": {name: sha256(path) for name, (_, path) in BASELINES.items()},
               "combined_frozen_touch_support_sha256": sha256(touch_path),
               "old_ownership_decoder_replay": replay,
               "evaluation_source_sha256": sha256(__file__),
               "external_run_metadata": metadata, "team_transition_cost": TEAM_COST,
               "fixed_live_dead_decisions": True, "source_video_extraction": False,
               "definitions": {
                   "primary_metric": "Official spellF1 mean at IoU 0.3,0.4,0.5,0.6,0.7; macroF1 first tie-breaker.",
                   "short_recovery": "Reference H/A spell <=125 frames (5s); official greedy same-team one-to-one IoU >=0.5.",
                   "evidence_scope": "NLL, binary Brier (p(A)-y)^2, and accuracy only on reference H/A; fixed-live and missing-ball subsets are explicit.",
                   "decoder": "Retained: unmodified binary_decode(cost=6). Combined: unmodified touch_decode(cost=6) with frozen destination touch support, switch costs3..6. Each decodes separately inside its own baseline H/A islands; D and X exactly fixed.",
                   "short_gain": "Maximum positive contiguous signed-logit sum; retained interior two-switch cost=12. Combined frozen destination support yields total two-switch cost6..12, so12 is a conservative bound only. This is a local diagnostic, not guaranteed spell matching; island edges and competing spans affect DP.",
                   "visual": "Existing native ball-support/scene-reliability probability tempering only; no claim of touch classification or corresponding external video supervision.",
                   "scope": "Experimental full-length CSVs; public development accuracy only. Unlabelled changes are not full-match accuracy."},
               "evidence": {}, "arms": {}}
    external = external_event_receipt(args.run, metadata, pm)
    json_write(args.out / "external_event_receipt.json", external)
    results["external_event_summary"] = external["aggregate"]
    probabilities = {"temporal_raw": new["raw_owner"][:, 1], "temporal_calibrated": new["owner"][:, 1],
                     "temporal_native_gate": sigmoid(new["visual_logit"])}
    for name, old in cached.items():
        probabilities[f"{name}_contextual_before_contact"] = old["p_owner"][:, 1]
        probabilities[f"{name}_interaction"] = sigmoid(old["interaction_logit"])
    for name, probability in probabilities.items():
        scopes = {"all_reference_HA": mask, "ball_missing_reference_HA": mask & ~ball_seen,
                  "ball_observed_reference_HA": mask & ball_seen}
        scopes.update({base + "_fixed_live_reference_HA": mask & np.isin(codes, [pm.H, pm.A])
                       for base, codes in baseline_codes.items()})
        results["evidence"][name] = {scope: evidence_metrics(gt, probability, eligible) for scope, eligible in scopes.items()}

    arm_specs = {name: (name, codes, BASELINES[name][0], "baseline") for name, codes in baseline_codes.items()}
    arm_specs.update(predictions)
    for name, (base_name, codes, path, logit_key) in arm_specs.items():
        base = baseline_codes[base_name]
        logit = cached[base_name]["interaction_logit"] if logit_key == "baseline" else new[logit_key]
        official = evaluate_submission(path)
        aux = diagnostic(pm, gt, codes, {"interaction_logit": logit, "p_dead": cached[base_name]["p_dead"]})
        # Do not misname temporal outputs as physical-contact evidence.
        for event in aux["short_events"]:
            event["favorable_ownership_evidence_frames"] = event.pop("favorable_contact_frames")
            event["signed_ownership_logit_sum"] = event.pop("signed_contact_logit_sum")
        metrics = official["metrics"]
        arm = {key: metrics[key] for key in ["spell_f1_mean", "macro_f1", "per_class", "n_gt_spells", "n_pred_spells"]}
        arm.update(aux)
        arm.update({"baseline": base_name, "csv_path": str(path.resolve()), "csv_sha256": sha256(path),
                    "experimental": logit_key != "baseline", "public_changes": frame_changes(gt, base, codes, mask),
                    "changes_versus_retained": frame_changes(gt, baseline_codes["retained"], codes, mask),
                    "changed_unlabelled_frames": int(((codes != base) & ~mask).sum()),
                    "D_frame_changes_full": int(((codes == pm.D) != (base == pm.D)).sum()),
                    "X_frame_changes_full": int(((codes == pm.X) != (base == pm.X)).sum()),
                    "possession_effects": possession_effects(pm, gt, base, codes),
                    "onset_restart_pair_changes": fixed_boundary_changes(pm, gt, base, codes),
                    "short_evidence": short_evidence(pm, gt, logit, base, destination_support=base_name == "combined")})
        results["arms"][name] = arm
        json_write(args.out / (name + "_official.json"), official)
        print(json.dumps({"arm": name, **{k: arm[k] for k in ["spell_f1_mean", "macro_f1", "short_matched"]},
                          **arm["public_changes"]}), flush=True)
    for base_name in baseline_codes:
        decoded, argmax = [results["arms"][base_name + "_" + kind] for kind in ["decoded", "argmax"]]
        d, a = [predictions[base_name + "_" + kind][1] for kind in ["decoded", "argmax"]]
        results["arms"][base_name + "_decoded"]["decoder_versus_argmax"] = {
            **frame_changes(gt, a, d, mask),
            "argmax_short_matched": argmax["short_matched"], "decoded_short_matched": decoded["short_matched"],
            "possession_effects": possession_effects(pm, gt, a, d)}
    after = {name: sha256(ROOT / name) for name in expected}
    if after != before:
        raise AssertionError("Preserved candidates changed during evaluation")
    results["candidate_hashes_after"] = after
    results["candidate_preservation_verified"] = True
    json_write(args.out / "comparison.json", results)
    shutil.copyfile(__file__, args.out / "evaluator_source.py")
    summary = ["Public development ownership comparison; D/X fixed within each named baseline.", "",
               "| Arm | spellF1 | macroF1 | Short / 6 | Corrected | Damaged | Lost / recovered spells |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for name, arm in results["arms"].items():
        change, effect = arm["public_changes"], arm["possession_effects"]
        summary.append(f"| {name} | {arm['spell_f1_mean']:.6f} | {arm['macro_f1']:.6f} | {arm['short_matched']} | "
                       f"{change['corrected_frames']} | {change['newly_damaged_frames']} | "
                       f"{effect['matched_lost']} / {effect['matched_recovered']} |")
    summary += ["", "Corrected/damaged and lost/recovered are relative to the arm's own retained or combined baseline.",
                "All experimental CSVs are separate, validated, full length, and do not replace either preserved candidate.",
                "Public scores do not establish full-match accuracy. See comparison.json for external calibration provenance, evidence metrics, decoder losses, and per-possession effects."]
    (args.out / "summary.md").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
