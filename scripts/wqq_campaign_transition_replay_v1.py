"""Reproduce the completed transition-scale candidate from raw observations.

No fitting, scoring, selection or nomination is provided. Four fresh WPY views
must pass byte-exact comparison before the existing completed candidate permits
raw WQQ replay. Expected predictions are opened only after fresh match seals.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import wqq_campaign_raw_replay_v2 as chain
from scripts import wqq_campaign_transition_scale_v1 as scale

np, base = chain.np, chain.base
KIND = "wqq-campaign-completed-transition-raw-replay-v1"
DEFAULT_WORK = ROOT / "runs/wqq-090-20261031/transition-raw-replay-v1"
STUDY_NAME = "layer6-v70-transition-scale"
STUDY = ROOT / "runs/wqq-090-20261031/studies" / STUDY_NAME
HALVES, CONDITIONS = chain.HALVES, chain.CONDITIONS
POLICY = {"alpha": .5, "complexity": 2, "id": "s050_joint_a050", "parent_confidence": .98,
          "status_mode": "joint", "transition_scale": .5, "unchanged": False}
CANDIDATE = {"path": "reports/wqq-090-20261031/candidates/transition-scale-v1.json",
             "sha256": "be0b35633c7097997f3b3a9d365e39df697b0d36c59e65192c35d7285e320c2d"}
FOUNDATION = {
    "source": {"path": "scripts/wqq_campaign_raw_replay_v2.py",
               "sha256": "7444c4eb45696dfdc670021f07c6d574f04c1b5a541d9ff346d1153964d62b29"},
    "tests": {"path": "tests/test_wqq_campaign_raw_replay_v2.py",
              "sha256": "49bdf36647f0558d2a4f438f98bf5e3462de91099d26eda0d5bd735b256b4899"},
    "protocol": {"path": "runs/wqq-090-20261031/raw-replay-v2/protocol.json",
                 "sha256": "ac6473a7af84a2584be7225ce89960a93634cd238bc0833dfd7d07e684d368dc"},
    "comparison": {"path": "runs/wqq-090-20261031/raw-replay-v2/comparison.json",
                   "sha256": "bc1444acef69a463cd780ed6a629c0de993020c9858cd25666b1260e1515b363"},
    "fresh_seal": {"path": "runs/wqq-090-20261031/raw-replay-v2/fresh-seal.json",
                   "sha256": "eb991d9fc1375b3c4f6831995ebfcb4ca544d6cf08cd6180cfc2ca84d35bc05c"},
}


def metadata(ref):
    return chain.read(chain.checked(ref))


def keys(match):
    if match not in ("J03WPY", "J03WQQ"):
        raise PermissionError("Only the two registered replay matches exist")
    return [f"{match}-{h}-{c}" for h in HALVES for c in CONDITIONS]


def sources():
    result = {**chain.source_closure(), **scale.source_pins()}
    for p in (Path(__file__), ROOT / "tests/test_wqq_campaign_transition_replay_v1.py"):
        result[chain.relative(p)] = chain.digest(p)
    return result


def foundation():
    for ref in FOUNDATION.values():
        chain.checked(ref)
    protocol = chain.protocol_for(chain.checked(FOUNDATION["protocol"]).parent)
    report, seal = metadata(FOUNDATION["comparison"]), metadata(FOUNDATION["fresh_seal"])
    if (report["kind"] != chain.KIND + "-comparison" or report["all_arrays_bit_exact"] is not True
            or report["reference_reads"] or report["metric_computation"] is not False
            or report["fresh_seal"] != FOUNDATION["fresh_seal"]
            or seal["protocol"] != FOUNDATION["protocol"] or set(seal["views"]) != set(keys("J03WPY"))
            or set(report["views"]) != set(keys("J03WPY"))
            or any(not row["bit_exact"] or row["actual"] != row["expected"]
                   for view in report["views"].values() for row in view.values())):
        raise PermissionError("The exact raw-v2 WPY foundation is not accepted")
    return protocol


def completed_candidate():
    """Read lifecycle/identity metadata only; never consult metric values."""
    candidate = metadata(CANDIDATE)
    if (candidate.get("kind") != "unpromoted_research_candidate" or candidate.get("study") != STUDY_NAME
            or candidate.get("family") != "hierarchical-joint-v70-transition-scale"
            or candidate.get("one_frozen_nomination") is not True
            or candidate.get("prediction_sealed_before_reference_audit") is not True
            or candidate.get("WQQ_fitted_or_calibrated") is not False
            or candidate.get("independent_score_verification") is not True):
        raise PermissionError("Only the exact completed frozen candidate can be reproduced")
    refs = {Path(r["path"]).name: r for r in candidate["artifacts"]}
    if len(refs) != len(candidate["artifacts"]):
        raise ValueError("Ambiguous candidate artifact inventory")
    for ref in refs.values():
        chain.checked(ref)
    protocol, nomination = metadata(refs["protocol.json"]), metadata(refs["transition-scale-v1.json"])
    oof, admission = metadata(refs["oof-audit.json"]), metadata(refs["full-admission.json"])
    full, seal = metadata(refs["full-audit.json"]), metadata(refs["full-seal.json"])
    selection = metadata(refs["selection.json"])
    pp, nom = refs["protocol.json"], refs["transition-scale-v1.json"]
    if (protocol["kind"] != scale.KIND or protocol["candidate_id"] != STUDY_NAME
            or protocol["recipe"]["model_family"] != "v70" or protocol["recipe"]["policies"] != scale.policy_grid()
            or nomination["kind"] != "wqq-campaign-nomination-v1" or nomination["eligible"] is not True
            or nomination["candidate_id"] != STUDY_NAME or nomination["selected_run_id"] != STUDY_NAME
            or nomination["protocol"] != pp or nomination["oof_audit"] != refs["oof-audit.json"]
            or nomination["selected_recipe"] != protocol["recipe"]
            or nomination["recipe_sha256"] != base.canonical_recipe_hash(protocol["recipe"])
            or nomination["WQQ_reference_reads"]):
        raise PermissionError("Nomination does not bind this exact completed recipe")
    base.verify_sources(protocol["source_hashes"])
    model_protocol = metadata(protocol["recipe"]["model_protocol"])
    base.verify_sources(model_protocol["sources"])
    if (oof["kind"] != "wqq-campaign-audit-v1" or oof["stage"] != "oof"
            or oof["eligible"] is not True or oof["run_id"] != STUDY_NAME or oof["protocol"] != pp
            or admission["kind"] != "wqq-campaign-admission-v1" or admission["nomination"] != nom
            or admission["protocol"] != pp or admission["oof_seal"] != oof["prediction_seal"]
            or admission["reference_reads"]):
        raise PermissionError("The completed OOF/admission gate differs")
    if (full["kind"] != "wqq-campaign-audit-v1" or full["stage"] != "full"
            or full["run_id"] != STUDY_NAME or full["protocol"] != pp or full["nomination"] != nom
            or full["prediction_seal"] != refs["full-seal.json"]
            or full["WQQ_fitted_or_calibrated"] is not False
            or seal["kind"] != "wqq-campaign-prediction-seal-v1" or seal["stage"] != "full"
            or seal["run_id"] != STUDY_NAME or seal["protocol"] != pp or seal["nomination"] != nom
            or seal["reference_reads"] or set(f"{r['match']}-{r['half']}-{r['condition']}" for r in seal["rows"]) != set(keys("J03WQQ"))):
        raise PermissionError("A complete matching full seal and audit are required")
    if (selection["kind"] != "wqq-campaign-calibration-v1" or selection["run_id"] != STUDY_NAME
            or selection["protocol"] != pp or selection["producer_held"] != "full"
            or selection["selected"] != POLICY or selection["selected_id"] != POLICY["id"]
            or selection["policy_shared_across_conditions"] is not True or selection["calibration_matches"] != ["J03WPY"]
            or any(r["policy_selection"] != refs["selection.json"] for r in seal["rows"])):
        raise PermissionError("Only the existing shared s050_joint_a050 policy is permitted")
    cal_seal = metadata(selection["prediction_seal"])
    if (cal_seal["stage"] != "calibration" or cal_seal["protocol"] != pp
            or cal_seal["run_id"] != STUDY_NAME or cal_seal["producer_held"] != "full"
            or cal_seal["reference_reads"] or set(f"{r['match']}-{r['half']}-{r['condition']}" for r in cal_seal["rows"]) != set(keys("J03WPY"))):
        raise PermissionError("Full-producer WPY selection inputs are not completely sealed")
    parent_admission = metadata(refs["parent-cache-admission-v1.json"])
    repair = metadata(refs["parent-cache-repair-v1.json"])
    if (parent_admission["scientific_change"] is not False or parent_admission["reference_reads"]
            or parent_admission["repair_receipt"] != refs["parent-cache-repair-v1.json"]
            or parent_admission["admission"]["protocol"] != pp or parent_admission["admission"]["nomination"] != nom
            or repair["study_protocol"] != pp or repair["reference_reads"]
            or repair["transform"]["scientific_change"] is not False):
        raise PermissionError("Historical parent cache repair is not the admitted unchanged producer")
    return {"candidate": CANDIDATE, "artifacts": refs, "calibration_seal": selection["prediction_seal"],
            "policy": POLICY, "reference_arrays_opened": [], "metric_values_used": False}, {"J03WPY": cal_seal, "J03WQQ": seal}


def comparison_inventory(spec, gate, seals):
    """Capture expected array pins from receipts; do not open array payloads."""
    fixtures, receipts = {}, {}
    for match in ("J03WPY", "J03WQQ"):
        sealed = {f"{r['match']}-{r['half']}-{r['condition']}": r for r in seals[match]["rows"]}
        for key in keys(match):
            _, half, condition = key.split("-")
            vp = STUDY / "views/full" / (key + ".json")
            view_pin = chain.pin(vp); receipts[view_pin["path"]] = view_pin
            view = metadata(view_pin); ident = view["identity"]
            refs = ident["input_identity"]
            native, parent = metadata(refs["logit_receipt"]), metadata(refs["parent_receipt"])
            for item in (refs["logit_receipt"], refs["parent_receipt"]):
                receipts[item["path"]] = item
            expected_id = {"which": "full", "match": match, "half": half, "condition": condition}
            if (any(ident.get(k) != v or native.get(k) != v or parent.get(k) != v for k, v in expected_id.items())
                    or ident["protocol"] != gate["artifacts"]["protocol.json"] or view["reference_reads"]
                    or native["reference_reads"] or parent["reference_arrays_opened"]
                    or native["family"] != "v70" or native["permit"] != gate["artifacts"]["transition-scale-v1.json"]
                    or native["corruption_seed"] != (100 if condition == "corrupted" else None)
                    or native["output"] != refs["logits"] or native["raw_input"] != refs["raw_input"]
                    or parent["artifact"] != refs["parent"] or view["artifact"] != sealed[key]["artifact"]
                    or native["model_bundle"]["model"] != spec["models"]["native"]
                    or native["model_bundle"]["context_model"] != spec["models"]["bridge"]
                    or set(native["model_bundle"]["fit_matches"]) != set(base.TRAIN)
                    or set(parent["fit_matches"]) != set(base.TRAIN)
                    or not set(parent["calibration_matches"]) <= set(base.TRAIN) | {"J03WPY"}):
                raise ValueError("Expected cache identity, model, parent, observation or seal differs")
            final_array = POLICY["id"] if match == "J03WPY" else "candidate"
            wanted = POLICY if match == "J03WPY" else {**POLICY, "id": "candidate"}
            if (wanted not in ident["policies"] or view["arrays"].get(final_array) != final_array
                    or sealed[key]["arrays"].get(final_array) != final_array
                    or (match == "J03WQQ" and (ident["policies"] != [wanted] or parent["nomination"] != gate["artifacts"]["transition-scale-v1.json"]))):
                raise ValueError("Expected output does not bind the frozen selected policy")
            raw_pin = refs["raw_input"]
            if raw_pin["path"] != f"private/improvement/external/prepared/{match}-{half}.npz":
                raise PermissionError("Only the exact registered raw observation file is permitted")
            fixtures[key] = {"match": match, "half": half, "condition": condition, "raw_input": raw_pin,
                "expected": {"parent": {"artifact": refs["parent"], "array": "parent"},
                             "logits": {"artifact": refs["logits"], "array": "logits"},
                             "transition": {"artifact": view["artifact"], "array": final_array}}}
    return fixtures, list(receipts.values())


def prepare(work=DEFAULT_WORK):
    work = Path(work)
    if (work / "protocol.json").exists():
        return protocol_for(work)
    original = foundation()
    gate, seals = completed_candidate()
    fixtures, receipts = comparison_inventory(original["spec"], gate, seals)
    src = sources()
    if chain.compiled_loaders(src) != original["spec"]["compiled_kernels"]["loaders"]:
        raise ValueError("New adapter adds an unregistered compiled loader")
    result = {"kind": KIND, "foundation": FOUNDATION, "gate": gate, "policy": POLICY,
              "spec": original["spec"], "sources": src, "runtime": chain.runtime_identity(),
              "fixtures": fixtures, "comparison_receipts": receipts, "reference_reads": [],
              "new_candidate": False, "policy_selection": False,
              "sequence": "WPY fresh seal -> byte-exact comparison -> completed-candidate WQQ replay -> separate comparison"}
    chain.immutable_json(work / "protocol.json", result)
    return result


def protocol_for(work):
    p = chain.read(Path(work) / "protocol.json")
    if (p["kind"] != KIND or p["foundation"] != FOUNDATION or p["policy"] != POLICY
            or p["sources"] != sources() or p["runtime"] != chain.runtime_identity()
            or p["reference_reads"] or p["new_candidate"] is not False or p["policy_selection"] is not False
            or set(p["fixtures"]) != set(keys("J03WPY") + keys("J03WQQ"))):
        raise ValueError("Frozen replay protocol/source/runtime changed")
    original = foundation()
    gate, seals = completed_candidate()
    if p["spec"] != original["spec"] or p["gate"] != gate:
        raise PermissionError("Replay foundation or completed-candidate gate changed")
    fixtures, receipts = comparison_inventory(p["spec"], gate, seals)
    if p["fixtures"] != fixtures or p["comparison_receipts"] != receipts:
        raise ValueError("Frozen raw/expected-array receipt bindings changed")
    for ref in p["comparison_receipts"]:
        chain.checked(ref)
    chain.verify_kernel_inventory(p["spec"])
    return p


class ReadBoundary(chain.ReadBoundary):
    """Exact allowlist; only registered raw WQQ files gain inference access."""
    def __init__(self, reads, match, *, comparison=False):
        keys(match)
        super().__init__(reads, comparison=comparison)
        self.match = match

    def __call__(self, event, args):
        if not self.active:
            return
        if event in ("subprocess.Popen", "os.system"):
            raise PermissionError("Replay cannot launch subprocesses")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        s = str(p)
        if (any(x in s for x in ("J03WR9", ".truth", ".actor", "/evaluation/gt/", "/runs/main/"))
                or "audit" in p.name.lower() or p.name.lower() in ("verdict.json", "best.json")):
            raise PermissionError("Reference/audit/reserved input is closed")
        mode, flags = args[1:3]
        if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))):
            raise PermissionError("Scientific replay cannot write files")
        raw_paths = {ROOT / f"private/improvement/external/prepared/{self.match}-{h}.npz" for h in HALVES}
        if "J03WQQ" in s and not (self.match == "J03WQQ" and (self.comparison or p in raw_paths)):
            raise PermissionError("WQQ access is limited to the admitted raw fixture or sealed comparison")
        if not self.comparison and p.suffix in (".npz", ".npy") and p not in raw_paths:
            raise PermissionError("Saved prediction/feature arrays are closed during inference")
        if p in self.reads:
            self.opened.add(p)
            return
        runtime = any(p.is_relative_to(ROOT / x) for x in (".venv", "private/improvement/runtime", "private/improvement/vision-env"))
        system = any(s.startswith(x) for x in ("/System/", "/usr/", "/Library/", "/opt/"))
        if (runtime or system) and p.suffix in (".py", ".pyc", ".so", ".dylib"):
            return
        raise PermissionError("Unlisted replay read: " + s)


def fixture_raw(row):
    match, half, condition = (row[k] for k in ("match", "half", "condition"))
    if f"{match}-{half}-{condition}" not in keys(match):
        raise PermissionError("Unknown replay fixture")
    if row["raw_input"]["path"] != f"private/improvement/external/prepared/{match}-{half}.npz":
        raise PermissionError("Raw path is not the registered physical half")
    with np.load(chain.checked(row["raw_input"]), allow_pickle=False) as z:
        if set(z.files) != chain.RAW_NAMES:
            raise ValueError("Input must contain exactly the four raw observation fields")
        raw = {k: z[k].copy() for k in sorted(chain.RAW_NAMES)}
    if condition == "corrupted":
        raw = chain.retained.corruption(raw, 100)
    chain.validate_raw(raw)
    return raw


def predict_raw(raw, bundle):
    """The scientific API has no match, condition, receipt or policy argument."""
    before = chain.raw_identity(raw)
    parent, intermediates, trace = bundle.parent_from_raw(raw)
    logits = bundle.native_from_raw(raw)
    result = scale.compose(parent, logits, POLICY)
    if chain.raw_identity(raw) != before:
        raise ValueError("Prediction mutated observations")
    return {"parent": parent, "logits": logits, "transition": result, **intermediates}, trace


def identity(protocol_pin, row, raw, admission):
    return {"protocol": protocol_pin, "match": row["match"], "half": row["half"], "condition": row["condition"],
            "raw_input": row["raw_input"], "observation_identity": chain.raw_identity(raw), "policy": POLICY,
            "corruption_seed": 100 if row["condition"] == "corrupted" else None, "admission": admission}


def committed(path, expected):
    m = chain.read(path)
    if (m["kind"] != KIND + "-fresh-view" or m["identity"] != expected
            or m["reference_reads"] or m["cached_prediction_reads"]):
        raise ValueError("Committed replay identity changed")
    chain.checked(m["output"])
    return m


def commit(work, key, ident, arrays, trace):
    folder = Path(work) / "fresh"; path = folder / (key + ".json")
    if path.exists():
        return committed(path, ident)
    folder.mkdir(parents=True, exist_ok=True)
    payload = folder / (key + "." + uuid.uuid4().hex + ".npz")
    with payload.open("xb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    result = {"kind": KIND + "-fresh-view", "identity": ident, "output": chain.pin(payload),
              "arrays": {k: chain.array_pin(v) for k, v in arrays.items()}, "trace": trace,
              "reference_reads": [], "cached_prediction_reads": []}
    chain.immutable_json(path, result)
    return result


def wqq_admission(work, protocol):
    """Completed candidate alone is insufficient: this adapter's WPY must pass."""
    path = Path(work) / "comparison-J03WPY.json"
    r = chain.read(path)
    seal_path = Path(work) / "fresh-J03WPY-seal.json"
    if (r["kind"] != KIND + "-comparison" or r["match"] != "J03WPY"
            or r["protocol"] != chain.pin(Path(work) / "protocol.json")
            or r["fresh_seal"] != chain.pin(seal_path) or r["all_arrays_bit_exact"] is not True
            or r["reference_reads"] or r["metric_computation"] is not False
            or set(r["views"]) != set(keys("J03WPY"))):
        raise PermissionError("All four fresh WPY views must pass exact comparison first")
    seal = chain.read(seal_path)
    if (seal["kind"] != KIND + "-fresh-seal" or seal["match"] != "J03WPY"
            or seal["admission"] is not None or seal["reference_reads"]
            or seal["protocol"] != r["protocol"] or set(seal["views"]) != set(keys("J03WPY"))):
        raise PermissionError("WPY comparison has another fresh seal")
    for key, results in r["views"].items():
        row = protocol["fixtures"][key]
        m = metadata(seal["views"][key]); chain.checked(m["output"])
        ident = m["identity"]
        if (m["kind"] != KIND + "-fresh-view" or m["reference_reads"] or m["cached_prediction_reads"]
                or ident["protocol"] != r["protocol"] or ident["policy"] != POLICY
                or ident["admission"] is not None or ident["raw_input"] != row["raw_input"]
                or any(ident[k] != row[k] for k in ("match", "half", "condition"))
                or ident["corruption_seed"] != (100 if row["condition"] == "corrupted" else None)):
            raise PermissionError("WPY comparison contains a misrouted fresh receipt")
        if set(results) != set(row["expected"]):
            raise PermissionError("WPY comparison omitted a required array")
        for name, result in results.items():
            if result["bit_exact"] is not True or result["actual"] != result["expected"] or result["actual"] != m["arrays"][name]:
                raise PermissionError("WPY parent/native/final byte parity is incomplete")
            chain.checked(row["expected"][name]["artifact"])
        chain.checked(row["raw_input"])
    gate, _ = completed_candidate()
    if gate != protocol["gate"]:
        raise PermissionError("The completed candidate changed after the WPY pilot")
    return {"candidate": CANDIDATE, "nomination": gate["artifacts"]["transition-scale-v1.json"],
            "completed_full_audit": gate["artifacts"]["full-audit.json"], "WPY_comparison": chain.pin(path)}


def predict(work=DEFAULT_WORK, match="J03WPY", half=None, condition=None):
    work = Path(work); p = protocol_for(work)
    all_keys = keys(match)
    admission = wqq_admission(work, p) if match == "J03WQQ" else None
    pp = chain.pin(work / "protocol.json")
    selected = [k for k in all_keys if (half is None or p["fixtures"][k]["half"] == half)
                and (condition is None or p["fixtures"][k]["condition"] == condition)]
    if not selected:
        raise ValueError("No registered replay views selected")
    bundle = None
    for key in selected:
        row = p["fixtures"][key]
        reads = [ROOT / s for s in p["sources"]] + [ROOT / r["path"] for r in p["spec"]["runtime_pins"]] + [ROOT / row["raw_input"]["path"]]
        with ReadBoundary(reads, match):
            raw = fixture_raw(row)
            ident = identity(pp, row, raw, admission)
        receipt = work / "fresh" / (key + ".json")
        if receipt.exists():
            committed(receipt, ident)
            print("TRANSITION REPLAY RESUME", key, flush=True)
            continue
        with ReadBoundary(reads, match):
            if bundle is None:
                bundle = chain.load_bundle(p["spec"])
            arrays, trace = predict_raw(raw, bundle)
        commit(work, key, ident, arrays, trace)
        print("TRANSITION REPLAY COMMITTED", key, flush=True)
    if all((work / "fresh" / (k + ".json")).exists() for k in all_keys):
        chain.immutable_json(work / f"fresh-{match}-seal.json", {
            "kind": KIND + "-fresh-seal", "protocol": pp, "match": match, "admission": admission,
            "views": {k: chain.pin(work / "fresh" / (k + ".json")) for k in all_keys}, "reference_reads": []})


def compare(work=DEFAULT_WORK, match="J03WPY"):
    work = Path(work); p = protocol_for(work)
    admission = wqq_admission(work, p) if match == "J03WQQ" else None
    seal_path = work / f"fresh-{match}-seal.json"
    seal = chain.read(seal_path); pp = chain.pin(work / "protocol.json")
    if (seal["kind"] != KIND + "-fresh-seal" or seal["protocol"] != pp or seal["match"] != match
            or seal["admission"] != admission or seal["reference_reads"] or set(seal["views"]) != set(keys(match))):
        raise PermissionError("All four fresh outputs must commit before expected arrays open")
    views = {}
    for key in keys(match):
        row = p["fixtures"][key]
        with ReadBoundary([ROOT / row["raw_input"]["path"]], match):
            raw = fixture_raw(row)
        views[key] = committed(chain.checked(seal["views"][key]), identity(pp, row, raw, admission))
    reads = [ROOT / v["output"]["path"] for v in views.values()]
    reads += [ROOT / target["artifact"]["path"] for k in views for target in p["fixtures"][k]["expected"].values()]
    results = {}
    with ReadBoundary(reads, match, comparison=True):
        for key, m in views.items():
            with np.load(chain.checked(m["output"]), allow_pickle=False) as z:
                actual = {k: z[k].copy() for k in z.files}
            if {k: chain.array_pin(v) for k, v in actual.items()} != m["arrays"]:
                raise ValueError("Fresh payload array manifest differs")
            results[key] = {}
            for name, target in p["fixtures"][key]["expected"].items():
                with np.load(chain.checked(target["artifact"]), allow_pickle=False) as z:
                    results[key][name] = chain.compare_arrays(actual[name], z[target["array"]])
    result = {"kind": KIND + "-comparison", "protocol": pp, "match": match, "fresh_seal": chain.pin(seal_path),
              "all_arrays_bit_exact": all(v["bit_exact"] for rows in results.values() for v in rows.values()),
              "views": results, "reference_reads": [], "metric_computation": False, "new_candidate": False}
    chain.immutable_json(work / f"comparison-{match}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict-wpy", "compare-wpy", "predict-wqq", "compare-wqq"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--half", choices=HALVES)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    if not args.stage.startswith("predict-") and (args.half or args.condition):
        parser.error("Only prediction stages accept view filters")
    if args.stage == "prepare":
        prepare(args.work)
        print(chain.pin(args.work / "protocol.json"))
    elif args.stage.startswith("predict-"):
        predict(args.work, "J03WPY" if args.stage == "predict-wpy" else "J03WQQ", args.half, args.condition)
    else:
        result = compare(args.work, "J03WPY" if args.stage == "compare-wpy" else "J03WQQ")
        print({"match": result["match"], "all_arrays_bit_exact": result["all_arrays_bit_exact"]})
        if not result["all_arrays_bit_exact"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
