"""Nine-policy status-mass ablation using frozen v70 logits and parent priors.

Five controls are exact original scalar mixtures. Four new policies preserve
their H/A conditional odds while replacing only the D marginal by the parent
prior D mass. Typed hazards and decoding are unchanged; no hard status lock or
condition flag is introduced. Only exact OOF admission permits full inference.
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import wqq_campaign_gate as gate

base, study, runtime = gate.base, gate.study, gate.runtime
np, torch = base.np, base.torch
DEFAULT_SOURCE = gate.DEFAULT_SOURCE
DEFAULT_STUDY = study.CAMPAIGN / "studies/layer5-v70-parent-status-mass"
DEFAULT_DIAGNOSTIC = study.CAMPAIGN / "diagnostics/v70-components-v1.json"
KIND = "wqq-campaign-parent-status-mass-study-v1"
Q = .98


def mechanism():
    return {"composition": "parent-D-marginal-preserved-v70-HA-odds-v1",
            "controls": "Exact base.compose for alpha=0,.25,.5,.75,1; alpha0 exact parent",
            "intervention": "From original clipped mixed log nodes L: new L_HA=L_HA-logsumexp(L_H,L_A)+log(1-parent_prior_D); new L_D=log(parent_prior_D)",
            "numeric": "Original float32 mixture and clipped log nodes, then float64 marginal replacement; no second probability clipping",
            "parent_X": "Same uniform 1/3 parent prior",
            "decoder": "Unchanged v70 typed hazards and emission/5; all states remain editable",
            "interpretation": "Soft status prior only; coupled decoding does not guarantee status-label identity"}


def policy_grid():
    controls = [{"id": f"joint_a{round(a*100):03d}", "alpha": a, "status_mode": "joint",
                 "parent_confidence": Q, "unchanged": a == 0., "complexity": 0} for a in base.ALPHAS]
    variants = [{"id": f"parentD_a{round(a*100):03d}", "alpha": a, "status_mode": "parent_D",
                 "parent_confidence": Q, "unchanged": False, "complexity": 1} for a in base.ALPHAS if a > 0]
    return controls + variants


def validate_policy(policy):
    canonical = [{k: v for k, v in p.items() if k != "id"} for p in policy_grid()]
    if {k: v for k, v in policy.items() if k != "id"} not in canonical:
        raise ValueError("Policy is outside the frozen nine-entry status-mass grid")
    if not isinstance(policy.get("id"), str) or not policy["id"]:
        raise ValueError("A stable named policy is required")


def mixed_nodes(parent, logits, alpha):
    """Original mixture arithmetic, retaining its existing clipping semantics."""
    parent, logits = np.asarray(parent), np.asarray(logits, np.float32)
    if (parent.ndim != 1 or parent.dtype.kind not in "iu" or not len(parent)
            or not np.isin(parent, [0, 1, 2, 3]).all() or logits.shape != (len(parent), 5)
            or not np.isfinite(logits).all() or alpha not in base.ALPHAS):
        raise ValueError("Canonical parent and aligned finite v70 logits required")
    tensor = torch.from_numpy(logits)
    native = base.emissions_api.emissions(tensor[:, 0], tensor[:, 1]).exp().numpy()
    prior = np.full_like(native, (1. - Q) / 2.)
    known = parent < 3
    prior[np.flatnonzero(known), parent[known].astype(int)] = Q
    prior[~known] = 1. / 3.
    return np.log(np.clip((1. - alpha) * prior + alpha * native, 1e-8, 1.)), prior


def parent_status_nodes(parent, logits, alpha):
    original, prior = mixed_nodes(parent, logits, alpha)
    nodes = original.astype(np.float64)
    parent_dead = prior[:, 2].astype(np.float64)
    live_logsum = np.logaddexp(nodes[:, 0], nodes[:, 1])
    nodes[:, :2] += (np.log1p(-parent_dead) - live_logsum)[:, None]
    nodes[:, 2] = np.log(parent_dead)
    if not np.isfinite(nodes).all():
        raise FloatingPointError("Invalid parent-status emission nodes")
    return nodes


def compose(parent, logits, policy):
    """Pure observation/model composition; no match or condition argument."""
    validate_policy(policy)
    if policy["status_mode"] == "joint":
        return base.compose(parent, logits, policy["alpha"], Q)
    nodes = parent_status_nodes(parent, logits, policy["alpha"])
    return base.native.decode(nodes, np.asarray(logits, np.float32)[:, 2:])


def source_pins():
    sources = gate.source_pins()
    for path in (Path(__file__), ROOT / "tests/test_wqq_campaign_status_mass_v1.py",
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


def register(folder=DEFAULT_STUDY, source=DEFAULT_SOURCE, diagnostic=DEFAULT_DIAGNOSTIC):
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
    index = {}
    for which in base.TRAIN:
        for match in (which, "J03WPY"):
            for half in base.HALVES:
                for condition in ("clean", "corrupted"):
                    index[gate.view_key(which, match, half, condition)] = gate._cache_entry(source, original_pin, which, match, half, condition)
    recipe = {"model_family": "v70", "model_protocol": original["recipe"]["model_protocol"],
              "mechanism": mechanism(), "policies": policy_grid(), "parent_confidence": Q,
              "parent": original["recipe"]["parent"], "conditions": ["clean", "corrupted"], "corruption_seed": 100,
              "selection": "One equal-condition full-match WPY policy per producer; exact ties macroF1, unchanged, original control, registry order",
              "diagnostic_motivation": diagnosis}
    protocol = {"kind": KIND, "candidate_id": folder.name, "recipe": recipe, "source_hashes": source_pins(),
                "source_study": base.relative(source), "source_protocol": original_pin,
                "source_oof_seal": base.pin(source / "oof-seal.json"), "cache_index": index,
                "train_matches": list(base.TRAIN), "calibration_matches": ["J03WPY"],
                "target_match": "J03WQQ", "target_role": "One exact eligible OOF nomination, then one full audit",
                "controls": ["v70", "v69"], "primary_metric": "Equal-condition official both-half five-IoU SpellF1",
                "cohort_selection": "Physical OOF mean; no policy refinement after held audit",
                "no_human_challenge_annotations": True}
    base.atomic_json(folder / "protocol.json", protocol, immutable=True)
    return protocol


def protocol_for(folder):
    protocol = study.protocol_for(Path(folder))
    recipe = protocol["recipe"]
    if (protocol["kind"] != KIND or recipe["model_family"] != "v70" or recipe["policies"] != policy_grid()
            or recipe["mechanism"] != mechanism() or recipe["parent_confidence"] != Q
            or recipe["conditions"] != ["clean", "corrupted"] or recipe["corruption_seed"] != 100):
        raise ValueError("Unregistered parent-status-mass protocol")
    for ref in (protocol["source_protocol"], protocol["source_oof_seal"], *recipe["diagnostic_motivation"].values()):
        base.checked(ref)
    base.verify_sources(base.read_json(base.checked(recipe["model_protocol"]))["sources"])
    return protocol


def produce_view(folder, which, match, half, condition, policies, permit=None):
    folder = Path(folder)
    protocol = protocol_for(folder)
    for policy in policies:
        validate_policy(policy)
    parent, logits, native, parent_meta, refs = gate._inputs(folder, protocol, which, match, half, condition, permit)
    identity = {"which": which, "match": match, "half": half, "condition": condition,
                "protocol": base.pin(folder / "protocol.json"), "input_identity": refs, "policies": policies}
    destination = folder / "views" / which / f"{match}-{half}-{condition}.npz"
    receipt = destination.with_suffix(".json")
    if receipt.exists():
        metadata = base.read_json(receipt)
        if metadata["identity"] != identity:
            raise ValueError("Status-mass composition resume identity mismatch")
        for ref in [metadata["artifact"], *metadata["inputs"]]:
            base.checked(ref)
        return metadata
    if destination.exists():
        raise FileExistsError("Unsealed status-mass output requires inspection")
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
    inputs = [*refs.values(), base.pin(folder / "protocol.json"), *protocol["recipe"]["diagnostic_motivation"].values()]
    output_name = "status-mass-" + base.pin(destination)["sha256"] + "-" + key
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


def full_audit(folder, permit):
    protocol_for(folder)
    admission = runtime.preflight_full(folder, permit)
    admission["status_mass_adapter"] = base.pin(Path(__file__))
    base.atomic_json(Path(folder) / "full-admission.json", admission, immutable=True)
    with installed():
        return study.full_audit(Path(folder), permit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "outer", "oof-audit", "full"))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--source-study", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--held", choices=base.TRAIN)
    parser.add_argument("--permit", type=Path)
    args = parser.parse_args()
    folder = args.study.resolve(); folder.relative_to(study.CAMPAIGN / "studies")
    base.threadpool_limits(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    sys.addaudithook(study.guard)
    try:
        if args.command == "register":
            register(folder, args.source_study, args.diagnostic)
            study.emit(stage="registered", study=folder.name, policies=9)
        else:
            protocol_for(folder)
            with installed():
                if args.command == "outer":
                    if args.held is None:
                        parser.error("outer requires --held")
                    study.produce_scored_match(folder, args.held, args.held)
                elif args.command == "oof-audit":
                    study.audit_oof(folder)
                else:
                    if args.permit is None:
                        parser.error("full requires --permit")
                    full_audit(folder, args.permit)
    except base.CheckpointedStop as exc:
        study.emit(stage="checkpointed", message=str(exc))
        raise SystemExit(75) from exc


if __name__ == "__main__":
    main()
