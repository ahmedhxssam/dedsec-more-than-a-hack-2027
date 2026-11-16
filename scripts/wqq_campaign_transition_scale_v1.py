"""One frozen 27-policy transition-energy sensitivity wave; no new fitting.

The nine status-mass policies cross global transition scales 1, .5 and 2.
Scale 1 calls the frozen composition exactly; alpha zero always returns the
parent. Other scales multiply all non-self energies, including restart log2,
with emission/5 unchanged. A sole WPY policy applies to both conditions.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import wqq_campaign_status_mass_v1 as status

gate = status.gate

base, study, runtime = gate.base, gate.study, gate.runtime
np, torch = base.np, base.torch
DEFAULT_SOURCE = gate.DEFAULT_SOURCE
DEFAULT_STUDY = study.CAMPAIGN / "studies/layer6-v70-transition-scale"
DEFAULT_STATUS = status.DEFAULT_STUDY
DEFAULT_DIAGNOSTIC = study.CAMPAIGN / "diagnostics/v70-components-v1.json"
KIND = "wqq-campaign-transition-scale-study-v1"
SCALES = (1., .5, 2.)
Q = .98


def mechanism():
    return {"composition": "global-transition-energy-scale-v1", "status_mass": status.mechanism(),
            "scales": list(SCALES), "controls": "Scale1 invokes exact frozen status.compose; alpha0 exact parent for every scale",
            "intervention": "Multiply entire off-diagonal transition energy by scale, INCLUDING D-to-live log2 destination prior; diagonals remain zero",
            "decoder": "Explicit float64 Viterbi with unchanged node/5, initialization and deterministic tie order; no hazard-logit multiplication",
            "limits": "No per-type scales, condition flags, offsets, caps, duration constraints or further grids",
            "gate": "Strict overall OOF gain over both v70 and frozen status-mass parent; v69 and every condition/pair regression reported without veto",
            "stop": "One frozen wave only; reject a failed scale hypothesis without refinement"}


def policy_grid():
    return [{**p, "id": f"s{round(scale*100):03d}_" + p["id"], "transition_scale": scale,
             "complexity": p["complexity"] + (0 if scale == 1 else 2)}
            for scale in SCALES for p in status.policy_grid()]


def validate_policy(policy):
    canonical = [{k: v for k, v in p.items() if k != "id"} for p in policy_grid()]
    if {k: v for k, v in policy.items() if k != "id"} not in canonical:
        raise ValueError("Policy is outside the frozen 27-entry transition-scale grid")
    if not isinstance(policy.get("id"), str) or not policy["id"]:
        raise ValueError("A stable named policy is required")


def transition_scores(hazard, scale):
    if scale not in SCALES:
        raise ValueError("Unregistered global transition scale")
    hazard = np.asarray(hazard, np.float64)
    if hazard.ndim != 2 or hazard.shape[1] != 3 or not np.isfinite(hazard).all():
        raise ValueError("Finite three-channel typed hazards required")
    cost = np.logaddexp(0., -hazard)
    f, d, s = cost.T
    zero = np.zeros(len(hazard), np.float64)
    return scale * np.stack([zero, -f, -d, -f, zero, -d,
                             -s-np.log(2.), -s-np.log(2.), zero], -1).reshape(-1, 3, 3)


def decode(nodes, hazard, scale):
    nodes = np.asarray(nodes, np.float64)
    transition = transition_scores(hazard, scale)
    if nodes.shape != (len(transition), 3) or not len(nodes) or not np.isfinite(nodes).all():
        raise ValueError("Aligned nonempty finite three-state nodes required")
    score = nodes[0] / 5.; back = np.zeros((len(nodes), 3), np.int8)
    states = np.arange(3)
    for i in range(1, len(nodes)):
        moved = score[:, None] + transition[i]
        back[i] = np.argmax(moved, axis=0)
        score = moved[back[i], states] + nodes[i] / 5.
    result = np.empty(len(nodes), np.int8); result[-1] = np.argmax(score)
    for i in range(len(nodes)-1, 0, -1):
        result[i-1] = back[i, result[i]]
    return result


def compose(parent, logits, policy):
    """Pure observation/model composition; no match or condition argument."""
    validate_policy(policy)
    original = next(p for p in status.policy_grid() if p["alpha"] == policy["alpha"] and p["status_mode"] == policy["status_mode"])
    if policy["transition_scale"] == 1 or policy["alpha"] == 0:
        return status.compose(parent, logits, original)
    nodes = (status.mixed_nodes(parent, logits, policy["alpha"])[0] if policy["status_mode"] == "joint"
             else status.parent_status_nodes(parent, logits, policy["alpha"]))
    return decode(nodes, np.asarray(logits, np.float32)[:, 2:], policy["transition_scale"])


def source_pins():
    sources = status.source_pins()
    for path in (Path(__file__), ROOT / "tests/test_wqq_campaign_transition_scale_v1.py",
                 ROOT / "scripts/wqq_campaign_error_diagnostic_v1.py"):
        sources[base.relative(path)] = base.sha256(path)
    return sources


def diagnostic_evidence(path):
    path = Path(path).resolve()
    report = base.read_json(path)
    expected = {(m, h) for m in base.TRAIN for h in base.HALVES}
    reads = report.get("reference_reads", [])
    if (report.get("kind") != "wqq-v70-error-diagnostic-v1" or report.get("diagnostic_only") is not True
            or report.get("fits_or_decodes") != 0 or len(reads) != 8
            or {(r.get("match"), r.get("half")) for r in reads} != expected):
        raise ValueError("A completed TRAIN4-only component diagnostic is required")
    sealed = base.read_json(base.checked(report["input_seal"]))
    if sealed.get("kind") != "wqq-v70-error-diagnostic-inputs-v1" or sealed.get("reference_reads") != []:
        raise ValueError("Component diagnostic inputs were not sealed before truth")
    for ref in sealed["inputs"]:
        # The diagnostic's input seal contains sources/observations/predictions,
        # never its later truth-read inventory.
        study.evaluation.checked(ref)
    return {"report": base.pin(path), "input_seal": report["input_seal"]}


def status_evidence(folder):
    """Bind only the completed TRAIN4 audit; never inspect the parent's full run."""
    folder = Path(folder).resolve()
    parent_protocol = status.protocol_for(folder)
    audit_path = folder / "oof-audit.json"
    audit = base.read_json(audit_path)
    protocol_pin, seal_pin = base.pin(folder / "protocol.json"), base.pin(folder / "oof-seal.json")
    if (audit.get("kind") != "wqq-campaign-audit-v1" or audit.get("stage") != "oof"
            or audit.get("run_id") != folder.name or audit.get("protocol") != protocol_pin
            or audit.get("prediction_seal") != seal_pin
            or set(audit.get("per_match", {})) != set(base.TRAIN)):
        raise ValueError("A complete exact status-mass TRAIN4 OOF audit is required")
    sealed, _ = study.evaluation.load_seal(folder / "oof-seal.json", stage="oof")
    if sealed["run_id"] != folder.name or sealed["protocol"] != protocol_pin:
        raise ValueError("Status-mass control seal identity mismatch")
    return {"protocol": protocol_pin, "oof_audit": base.pin(audit_path), "oof_seal": seal_pin}, parent_protocol


def register(folder=DEFAULT_STUDY, source=DEFAULT_SOURCE, diagnostic=DEFAULT_DIAGNOSTIC,
             status_folder=DEFAULT_STATUS):
    folder, source = Path(folder).resolve(), Path(source).resolve()
    folder.relative_to(study.CAMPAIGN / "studies"); source.relative_to(study.CAMPAIGN / "studies")
    if folder == source:
        raise ValueError("Preserve the existing source study")
    original = study.protocol_for(source)
    if original["recipe"]["model_family"] != "v70":
        raise ValueError("Only frozen v70 source logits are supported")
    original_pin = base.pin(source / "protocol.json")
    sealed, _ = study.evaluation.load_seal(source / "oof-seal.json", stage="oof")
    if sealed["run_id"] != source.name or sealed["protocol"] != original_pin:
        raise ValueError("Source OOF seal identity mismatch")
    diagnosis = diagnostic_evidence(diagnostic)
    parent_control, parent_protocol = status_evidence(status_folder)
    if folder == Path(status_folder).resolve():
        raise ValueError("Preserve the frozen status-mass parent")
    index = {}
    for which in base.TRAIN:
        for match in (which, "J03WPY"):
            for half in base.HALVES:
                for condition in ("clean", "corrupted"):
                    index[gate.view_key(which, match, half, condition)] = gate._cache_entry(source, original_pin, which, match, half, condition)
    if parent_protocol["source_protocol"] != original_pin or parent_protocol["cache_index"] != index:
        raise ValueError("Transition sensitivity and status-mass control must use identical native/parent inputs")
    recipe = {"model_family": "v70", "model_protocol": original["recipe"]["model_protocol"],
              "mechanism": mechanism(), "policies": policy_grid(), "parent_confidence": Q,
              "parent": original["recipe"]["parent"], "conditions": ["clean", "corrupted"], "corruption_seed": 100,
              "selection": "One equal-condition full-match WPY policy per producer; exact ties macroF1, unchanged, scale1, original status control, registry order",
              "diagnostic_motivation": diagnosis, "status_control": parent_control}
    protocol = {"kind": KIND, "candidate_id": folder.name, "recipe": recipe, "source_hashes": source_pins(),
                "source_study": base.relative(source), "source_protocol": original_pin,
                "source_oof_seal": base.pin(source / "oof-seal.json"), "cache_index": index,
                "train_matches": list(base.TRAIN), "calibration_matches": ["J03WPY"],
                "target_match": "J03WQQ", "target_role": "One exact eligible OOF nomination, then one full audit",
                "controls": ["v70", "v69", "status_mass"], "primary_metric": "Equal-condition official both-half five-IoU SpellF1",
                "cohort_selection": "Strict overall gain over v70 and status_mass; report all condition/pair regressions without veto; one wave without refinement",
                "no_human_challenge_annotations": True}
    base.atomic_json(folder / "protocol.json", protocol, immutable=True)
    return protocol


def protocol_for(folder):
    protocol = study.protocol_for(Path(folder))
    recipe = protocol["recipe"]
    if (protocol["kind"] != KIND or recipe["model_family"] != "v70" or recipe["policies"] != policy_grid()
            or recipe["mechanism"] != mechanism() or recipe["parent_confidence"] != Q
            or recipe["conditions"] != ["clean", "corrupted"] or recipe["corruption_seed"] != 100):
        raise ValueError("Unregistered transition-scale protocol")
    for ref in (protocol["source_protocol"], protocol["source_oof_seal"],
                *recipe["diagnostic_motivation"].values(), *recipe["status_control"].values()):
        base.checked(ref)
    base.verify_sources(base.read_json(base.checked(recipe["model_protocol"]))["sources"])
    return protocol


def produce_view(folder, which, match, half, condition, policies, permit=None):
    folder = Path(folder)
    protocol = protocol_for(folder)
    for policy in policies:
        validate_policy(policy)
    if which == "full":
        preflight_full(folder, permit)
    parent, logits, native, parent_meta, refs = gate._inputs(folder, protocol, which, match, half, condition, permit)
    identity = {"which": which, "match": match, "half": half, "condition": condition,
                "protocol": base.pin(folder / "protocol.json"), "input_identity": refs, "policies": policies}
    destination = folder / "views" / which / f"{match}-{half}-{condition}.npz"
    receipt = destination.with_suffix(".json")
    if receipt.exists():
        metadata = base.read_json(receipt)
        if metadata["identity"] != identity:
            raise ValueError("Transition-scale composition resume identity mismatch")
        for ref in [metadata["artifact"], *metadata["inputs"]]:
            base.checked(ref)
        return metadata
    if destination.exists():
        raise FileExistsError("Unsealed transition-scale output requires inspection")
    base.atomic_npz(destination, **{p["id"]: compose(parent, logits, p) for p in policies})
    nodes = {}
    learned = [study.weight_nodes(a["freeze"], nodes) for a in native["model_bundle"]["learned_ancestors"]]
    parent_name = "parent-" + parent_meta["manifest"]["sha256"]
    nodes[parent_name] = {"kind": "historical_parent", "fit_matches": parent_meta["fit_matches"],
                          "calibration_matches": parent_meta["calibration_matches"], "dependencies": [],
                          "artifacts": [parent_meta["manifest"], parent_meta["artifact"], parent_meta["source"]],
                          "role_receipt": parent_meta["manifest"]}
    key = gate.view_key(which, match, half, condition)
    native_name = "logits-" + refs["logits"]["sha256"] + "-" + key
    nodes[native_name] = {"kind": "pipeline", "fit_matches": [], "calibration_matches": [], "dependencies": learned,
                          "artifacts": [refs["logit_receipt"], refs["logits"], refs["raw_input"],
                                        *[{"path": p, "sha256": s} for p, s in native["sources"].items()]]}
    inputs = [*refs.values(), base.pin(folder / "protocol.json"),
              *protocol["recipe"]["diagnostic_motivation"].values(), *protocol["recipe"]["status_control"].values()]
    output_name = "transition-scale-" + base.pin(destination)["sha256"] + "-" + key
    nodes[output_name] = {"kind": "pipeline", "fit_matches": [], "calibration_matches": [],
                          "dependencies": [parent_name, native_name], "artifacts": [base.pin(destination), *inputs]}
    metadata = {"kind": "wqq-campaign-composition-v1", "identity": identity, "artifact": base.pin(destination),
                "inputs": inputs, "ancestry_nodes": nodes, "ancestry_roots": [output_name],
                "reference_reads": [], "arrays": {p["id"]: p["id"] for p in policies}, "frames": len(parent)}
    base.atomic_json(receipt, metadata, immutable=True)
    study.emit(stage="prediction", study=folder.name, which=which, match=match, half=half,
               condition=condition, frames=len(parent), policies=len(policies), cached_native=which != "full")
    study.checkpoint(folder, base.relative(receipt))
    return metadata


@contextlib.contextmanager
def installed():
    original = study.produce_view
    study.produce_view = produce_view
    try:
        yield
    finally:
        study.produce_view = original


def attach_parent_control(result, parent, evidence):
    """Reuse immutable official parent metrics only for identical TRAIN4 truth."""
    result = copy.deepcopy(result)
    if (result.get("stage") != "oof" or parent.get("stage") != "oof"
            or result["reference_reads"] != parent["reference_reads"]
            or set(result["per_match"]) != set(base.TRAIN) or set(parent["per_match"]) != set(base.TRAIN)):
        raise ValueError("Candidate and status-mass control have different OOF reference identities")
    for key in ("overall", "macro_f1"):
        result[key]["status_mass"] = parent[key]["candidate"]
    for key in ("conditions", "per_match", "per_producer_pair"):
        if set(result[key]) != set(parent[key]):
            raise ValueError("Status-mass control coverage differs")
        for name in result[key]:
            result[key][name]["status_mass"] = parent[key][name]["candidate"]
    for match in base.TRAIN:
        for condition in ("clean", "corrupted"):
            result["details"][match][condition]["status_mass"] = copy.deepcopy(parent["details"][match][condition]["candidate"])
    result["control_seals"]["status_mass"] = evidence["oof_seal"]
    result["status_control"] = evidence
    result["delta_vs_status_mass"] = {
        "overall": result["overall"]["candidate"] - result["overall"]["status_mass"],
        **{key: {name: row["candidate"] - row["status_mass"] for name, row in result[key].items()}
           for key in ("conditions", "per_match", "per_producer_pair")}}
    result["status_control_comparison"] = "Immutable official parent audit on identical references; parent metrics and scalar deltas, no recomputed per-spell matching changes"
    result["eligible"] = bool(result["overall"]["candidate"] > max(result["overall"]["v70"], result["overall"]["status_mass"]))
    result["gate"] = mechanism()["gate"]
    result["one_wave_stop_rule"] = mechanism()["stop"]
    return result


def audit_oof(folder):
    folder = Path(folder)
    protocol = protocol_for(folder)
    evidence = protocol["recipe"]["status_control"]
    parent_folder = base.checked(evidence["protocol"]).parent
    actual, _ = status_evidence(parent_folder)
    if actual != evidence:
        raise ValueError("Registered status-mass control changed")
    parent = base.read_json(base.checked(evidence["oof_audit"]))
    seal, output = folder / "oof-seal.json", folder / "oof-audit.json"
    if output.exists():
        result = base.read_json(output)
        study.evaluation.load_seal(seal, stage="oof")
        if result["prediction_seal"] != base.pin(seal) or attach_parent_control(result, parent, evidence) != result:
            raise ValueError("Changed transition-scale audit evidence or gate")
        return result
    if not seal.exists():
        rows, nodes = [], {}
        for which in base.TRAIN:
            inventory = base.read_json(folder / "inventories" / f"{which}.json")
            if inventory["protocol"] != base.pin(folder / "protocol.json"):
                raise ValueError("Mixed study inventories")
            rows.extend(inventory["rows"]); study.add_nodes(nodes, inventory["ancestry_nodes"])
        study.evaluation.create_seal(seal, run_id=folder.name, stage="oof", rows=rows,
                                     policies=[{"id": "candidate"}], protocol=base.pin(folder / "protocol.json"), ancestry_nodes=nodes)
    with study.references(base.TRAIN):
        result = study.evaluation.audit_predictions(seal)
    result = attach_parent_control(result, parent, evidence)
    base.atomic_json(output, result, immutable=True)
    study.emit(stage="oof-audit", study=folder.name, overall=result["overall"], eligible=result["eligible"])
    return result


def preflight_full(folder, permit):
    if permit is None:
        raise PermissionError("Exact transition-scale nomination required before full inference")
    admission = runtime.preflight_full(folder, permit)
    nomination = base.read_json(permit)
    audit_path = Path(folder) / "oof-audit.json"
    if not audit_path.is_file() or nomination.get("oof_audit") != base.pin(audit_path):
        raise PermissionError("Full nomination must bind this study's passing v70/status-mass gate")
    audit = audit_oof(folder)  # Existing sealed audit verification only; no new truth access.
    if audit["eligible"] is not True:
        raise PermissionError("Full nomination must bind this study's passing v70/status-mass gate")
    return admission


def full_audit(folder, permit):
    protocol_for(folder)
    admission = preflight_full(folder, permit)
    admission["transition_scale_adapter"] = base.pin(Path(__file__))
    base.atomic_json(Path(folder) / "full-admission.json", admission, immutable=True)
    with installed():
        return study.full_audit(Path(folder), permit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "outer", "oof-audit", "full"))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--source-study", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--status-study", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--held", choices=base.TRAIN)
    parser.add_argument("--permit", type=Path)
    args = parser.parse_args()
    folder = args.study.resolve(); folder.relative_to(study.CAMPAIGN / "studies")
    base.threadpool_limits(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    sys.addaudithook(study.guard)
    try:
        if args.command == "register":
            register(folder, args.source_study, args.diagnostic, args.status_study)
            study.emit(stage="registered", study=folder.name, policies=27)
        else:
            protocol_for(folder)
            with installed():
                if args.command == "outer":
                    if args.held is None:
                        parser.error("outer requires --held")
                    study.produce_scored_match(folder, args.held, args.held)
                elif args.command == "oof-audit":
                    audit_oof(folder)
                else:
                    if args.permit is None:
                        parser.error("full requires --permit")
                    full_audit(folder, args.permit)
    except base.CheckpointedStop as exc:
        study.emit(stage="checkpointed", message=str(exc))
        raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
