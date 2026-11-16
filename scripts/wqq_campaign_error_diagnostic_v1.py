"""Bounded TRAIN4-only decomposition of already sealed v70 source/path errors.

No fit, decoder, policy search, WPY truth, full-match model or WQQ asset is used.
All 16 source views and selected candidate paths are verified before truth IO.
Run via the campaign runtime and light supervisor queue.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import wqq_campaign_runtime as runtime
from scripts import wqq_campaign_gate as gate
from scripts import wqq_campaign_evaluate as ev
from scripts import full_baseline_evaluate as historical

np, base, study = gate.np, gate.base, gate.study
DEFAULT_SOURCE = study.CAMPAIGN / "studies/layer1-v70-prior"
DEFAULT_OUT = study.CAMPAIGN / "diagnostics/v70-components-v1.json"
KINDS = ("ownership_flip", "live_to_dead", "dead_to_live")
TOLERANCES = (3, 12)
REGIONS = ("all", "observed25", "missing25")
RECIPE = {
    "version": 1, "matches": list(ev.TRAIN), "views": 16,
    "regions": "missing25 iff centered nearest 25-frame actual-ball missing fraction > 0; otherwise observed25",
    "head_decisions": "A iff owner logit > 0; D iff dead logit > 0; head errors evaluated on their conditional truth roles",
    "boundaries": "Official pair-typed one-to-one transition matches at +/-3 and +/-12 frames within each half and GT-valid run",
    "short_spells": "Official greedy matches at five IoUs; durations 1-25,26-125,over125; missing25 if any spell frame lies in missing25",
    "head_support": "For a missed GT spell at IoU .5, owner/status support separately means >=75% correct raw v70 head decisions over that spell",
    "hazards": "Mean softplus(-corrected typed logit) at true boundaries, separately matched/missed at tolerance12; restart excludes fixed log2 destination split",
    "limitations": "Adaptive TRAIN4 diagnosis, not causal attribution or independent accuracy. Candidate head support always refers to cached source v70 heads.",
}
_TRUTH_OPEN = False


def guard(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = os.fsdecode(args[0])
    if any(s in path for s in ("J03WQQ", "J03WR9", "/evaluation/gt/", "/runs/main/", ".actor.npy")):
        raise PermissionError("Diagnostic forbids reserved assets")
    if ".truth.npy" in path and (not _TRUTH_OPEN or Path(path).name.split("-")[0] not in ev.TRAIN):
        raise PermissionError("Only TRAIN4 truth after all source/path validation is allowed")


def add_counts(target, value):
    for key, item in value.items():
        if isinstance(item, dict):
            add_counts(target.setdefault(key, {}), item)
        else:
            target[key] = target.get(key, 0) + item


def regions(missing):
    return {"all": np.ones(len(missing), bool), "observed25": ~missing, "missing25": missing}


def frame_counts(truth, paths, logits, missing):
    owner, dead = (logits[:, 0] > 0).astype(np.int8), logits[:, 1] > 0
    parent = paths["parent"]
    result = {}
    for region, where in regions(missing).items():
        live, valid = where & (truth < 2), where & (truth < 3)
        pairs = {}
        for head, native_ok, parent_ok, eligible in (
                ("owner", owner == truth, parent == truth, live & (parent < 2)),
                ("dead", dead == (truth == 2), (parent == 2) == (truth == 2), valid & (parent < 3))):
            pairs[head] = {"eligible": int(eligible.sum()),
                           "both_correct": int((eligible & native_ok & parent_ok).sum()),
                           "native_only_correct": int((eligible & native_ok & ~parent_ok).sum()),
                           "parent_only_correct": int((eligible & ~native_ok & parent_ok).sum()),
                           "both_wrong": int((eligible & ~native_ok & ~parent_ok).sum())}
        row = {"graded": int(valid.sum()), "true_live": int(live.sum()),
               "native_owner_errors_on_true_live": int((live & (owner != truth)).sum()),
               "native_dead_errors_on_graded": int((valid & (dead != (truth == 2))).sum()),
               "parent_not_live_on_true_live": int((live & (parent >= 2)).sum()),
               "parent_native_complementarity": pairs, "paths": {}}
        for name, p in paths.items():
            row["paths"][name] = {
                "correct": int((valid & (p == truth)).sum()),
                "wrong_live_owner": int((live & (p < 2) & (p != truth)).sum()),
                "false_dead": int((live & (p == 2)).sum()),
                "false_live": int((valid & (truth == 2) & (p < 2)).sum()),
                "abstained": int((valid & (p == 3)).sum())}
        result[region] = row
    return result


def transition_kind(event):
    _, before, after = event
    if before < 2 and after < 2:
        return "ownership_flip"
    return "live_to_dead" if after == 2 else "dead_to_live"


def boundary_counts(pm, truth, paths, logits, missing):
    valid = pm.valid_runs(truth)
    actual = [t for t in pm.transitions(truth, valid) if t[1] < 3 and t[2] < 3]
    result = {}
    for name, labels in paths.items():
        predicted = [t for t in pm.transitions(pm.fill_abstentions(labels, valid), valid) if t[1] < 3 and t[2] < 3]
        result[name] = {}
        for kind_index, kind in enumerate(KINDS):
            gt, pred = ([t for t in events if transition_kind(t) == kind] for events in (actual, predicted))
            result[name][kind] = {}
            for tolerance in TOLERANCES:
                matched, missed, spurious = pm.match_transitions(gt, pred, tolerance, typed="pair")
                for region, where in regions(missing).items():
                    row = {"truth": sum(bool(where[t[0]]) for t in gt), "predicted": sum(bool(where[t[0]]) for t in pred),
                           "matched_truth": sum(bool(where[g[0]]) for g, _ in matched),
                           "matched_predicted": sum(bool(where[p[0]]) for _, p in matched),
                           "absolute_offset_sum": sum(abs(p[0] - g[0]) for g, p in matched if where[g[0]])}
                    if tolerance == 12:
                        for label, events in (("matched", [g for g, _ in matched]), ("missed", missed)):
                            times = [t[0] for t in events if where[t[0]]]
                            row[label + "_cost_count"] = len(times)
                            row[label + "_cost_sum"] = float(np.logaddexp(0., -logits[times, 2 + kind_index].astype(np.float64)).sum())
                    result[name][kind].setdefault(str(tolerance), {})[region] = row
    return result


def spell_counts(pm, truth, paths, logits, missing):
    valid = pm.valid_runs(truth)
    gt = pm.possession_spells(truth, valid)
    duration = np.asarray([b - a for a, b, _ in gt])
    gap = np.asarray([bool(missing[a:b].any()) for a, b, _ in gt], bool)
    owner = logits[:, 0] > 0
    owner_support = np.asarray([np.mean(owner[a:b] == team) >= .75 for a, b, team in gt], bool)
    status_support = np.asarray([np.mean(logits[a:b, 1] <= 0) >= .75 for a, b, _ in gt], bool)
    buckets = {"1_to_25": duration <= 25, "26_to_125": (duration > 25) & (duration <= 125), "over125": duration > 125}
    result, scores = {}, {}
    for name, labels in paths.items():
        predicted = pm.possession_spells(pm.fill_abstentions(labels, valid), valid)
        overlaps = pm.spell_overlaps(gt, predicted)
        flags = {str(t): pm.matched_gt_flags(overlaps, len(gt), t) for t in pm.SPELL_IOUS}
        denom = len(gt) + len(predicted)
        scores[name] = float(np.mean([2 * f.sum() / denom if denom else 0. for f in flags.values()]))
        result[name] = {}
        for region, where in regions(gap).items():
            for bucket, selected in buckets.items():
                mask = where & selected
                missed = mask & ~flags["0.5"]
                result[name].setdefault(region, {})[bucket] = {
                    "truth_spells": int(mask.sum()), "matched": {t: int((mask & f).sum()) for t, f in flags.items()},
                    "missed_at_05": int(missed.sum()),
                    "missed_owner_supported": int((missed & owner_support).sum()),
                    "missed_status_supported": int((missed & status_support).sum()),
                    "missed_both_heads_supported": int((missed & owner_support & status_support).sum())}
    return result, scores


def _verify_code(protocol):
    pins = []
    for field in ("source_hashes", "sources"):
        if isinstance(protocol.get(field), dict):
            pins.extend({"path": p, "sha256": s} for p, s in protocol[field].items())
    pins += protocol.get("code", [])
    if not pins:
        raise ValueError("Protocol must expose hash-bound source code")
    for pin in pins:
        ev.checked(pin)
    return pins


def run(source=DEFAULT_SOURCE, candidates=(), out=DEFAULT_OUT):
    global _TRUTH_OPEN
    source, out = ev._path(source), ev._path(out)
    out.relative_to(study.CAMPAIGN)
    if out.exists():
        raise FileExistsError("Diagnostic already exists; preserve its evidence")
    protocol = study.protocol_for(source)
    if protocol["recipe"]["model_family"] != "v70":
        raise ValueError("Diagnosis requires the frozen v70 source logits")
    source_protocol = ev.pin(source / "protocol.json")
    seal, source_cases = ev.load_seal(source / "oof-seal.json", stage="oof")
    if seal["run_id"] != source.name or seal["protocol"] != source_protocol or len(seal["policies"]) != 1:
        raise ValueError("Source study/seal mismatch")
    code = [Path(__file__), *(Path(module.__file__) for module in (gate, base, study, runtime, ev, historical))]
    evidence = [*(ev.pin(p) for p in code), source_protocol, ev.pin(source / "oof-seal.json"), *_verify_code(protocol)]
    named = {"source_selected": (source_cases, seal["policies"][0]["id"])}
    for candidate in candidates:
        folder = ev._path(candidate)
        candidate_seal, cases = ev.load_seal(folder / "oof-seal.json", stage="oof")
        p = ev.read(ev.checked(candidate_seal["protocol"]))
        if candidate_seal["run_id"] != folder.name or len(candidate_seal["policies"]) != 1:
            raise ValueError("Candidate identity/policy mismatch")
        if folder.name in named:
            raise ValueError("Duplicate candidate name")
        evidence += [ev.pin(folder / "oof-seal.json"), candidate_seal["protocol"], *_verify_code(p)]
        named[folder.name] = (cases, candidate_seal["policies"][0]["id"])
    controls, control_pins = ev._controls("oof", ev.TRAIN)
    old_protocol = ev.read(ROOT / "runs/hierarchical-joint-v70/protocol.json")
    evidence += [control_pins["v70"], ev.pin(ROOT / "runs/hierarchical-joint-v70/protocol.json"), *_verify_code(old_protocol)]
    loaded = {}
    for m in ev.TRAIN:
        for h in ev.HALVES:
            for c in ev.CONDITIONS:
                key = (m, h, c)
                entry = gate._cache_entry(source, source_protocol, m, m, h, c)
                evidence.extend(v for k, v in entry.items() if k != "identity")
                with np.load(ev.checked(entry["logits"]), allow_pickle=False) as z:
                    logits = z["logits"].copy()
                with np.load(ev.checked(entry["parent"]), allow_pickle=False) as z:
                    parent = ev.labels(z["parent"])
                raw, raw_pin = base.raw_fixture(m, h, c)
                if raw_pin != entry["raw_input"]:
                    raise ValueError("Diagnostic raw mask is not the logit source view")
                missing = gate.missing_fraction(gate.missing_mask(raw)) > 0
                del raw
                paths = {"parent": parent, "v70": controls["v70"][key],
                         **{name: cases[key][pid] for name, (cases, pid) in named.items()}}
                if (logits.shape != (len(parent), 5) or not np.isfinite(logits).all()
                        or len(missing) != len(parent) or any(len(p) != len(parent) for p in paths.values())):
                    raise ValueError("Misaligned diagnostic sources")
                loaded[key] = {"logits": logits, "paths": paths, "missing": missing}
    pins = {p["path"]: p for p in evidence}
    if any(p["sha256"] != pins[p["path"]]["sha256"] for p in evidence):
        raise ValueError("Conflicting source evidence pins")
    input_seal = {"kind": "wqq-v70-error-diagnostic-inputs-v1", "recipe": RECIPE,
                  "source_study": ev.pin(source / "protocol.json"), "inputs": [pins[k] for k in sorted(pins)],
                  "candidate_names": list(named), "reference_reads": []}
    seal_path = out.with_suffix(".inputs.json")
    if seal_path.exists():
        if ev.read(seal_path) != input_seal:
            raise ValueError("Diagnostic inputs changed since prior attempt")
    else:
        ev.write_new(seal_path, input_seal)
    try:
        _TRUTH_OPEN = True
        truths, reads = ev._truths(ev.TRAIN, source_cases)
    finally:
        _TRUTH_OPEN = False
    pm = ev.official_metrics()
    details, aggregates = {}, {}
    for m in ev.TRAIN:
        details[m] = {}
        for c in ev.CONDITIONS:
            views = [loaded[m, h, c] for h in ev.HALVES]
            truth_parts = [truths[m, h] for h in ev.HALVES]
            frames, boundaries = {}, {}
            for y, view in zip(truth_parts, views):
                add_counts(frames, frame_counts(y, view["paths"], view["logits"], view["missing"]))
                add_counts(boundaries, boundary_counts(pm, y, view["paths"], view["logits"], view["missing"]))
            y = historical.join_halves(truth_parts)
            paths = {name: historical.join_halves([v["paths"][name] for v in views]) for name in views[0]["paths"]}
            logits = np.concatenate([views[0]["logits"], np.zeros((1, 5), np.float32), views[1]["logits"]])
            missing = np.r_[views[0]["missing"], False, views[1]["missing"]]
            spells, scores = spell_counts(pm, y, paths, logits, missing)
            row = {"frames": frames, "boundaries": boundaries, "spells": spells, "official_spell_f1": scores}
            details[m][c] = row
            for scope in ("all_conditions", c):
                aggregate = aggregates.setdefault(scope, {"frames": {}, "boundaries": {}, "spells": {}})
                for category in ("frames", "boundaries", "spells"):
                    add_counts(aggregate[category], row[category])
    for scope, conditions in (("all_conditions", ev.CONDITIONS), *((c, (c,)) for c in ev.CONDITIONS)):
        aggregates[scope]["official_spell_f1"] = {name: float(np.mean([details[m][c]["official_spell_f1"][name]
                                for m in ev.TRAIN for c in conditions])) for name in paths}
    result = {"kind": "wqq-v70-error-diagnostic-v1", "input_seal": ev.pin(seal_path), "recipe": RECIPE,
              "reference_reads": reads, "details": details, "aggregates": aggregates, "diagnostic_only": True,
              "candidate_prediction_arrays_written": False, "fits_or_decodes": 0}
    ev.write_new(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidate-study", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    base.threadpool_limits(1); base.torch.set_num_threads(1)
    sys.addaudithook(guard)
    result = run(args.source_study, args.candidate_study, args.out)
    print(json.dumps({"kind": result["kind"], "scores": result["aggregates"]["all_conditions"]["official_spell_f1"]}), flush=True)


if __name__ == "__main__":
    main()
