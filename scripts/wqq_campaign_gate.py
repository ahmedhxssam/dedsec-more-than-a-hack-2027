"""Frozen missing25 routing of v70 three-state emissions against its parent.

Outer/WPY production consumes pinned layer1 caches without neural inference.
Only an exact eligible OOF nomination admits new full-model WPY/WQQ inference.
Fixture conditions construct observations; the composition API accepts only an
observation-derived missing mask, logits, parent labels and endpoint policy.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import wqq_campaign_runtime as runtime
from scripts import wqq_campaign_model as base
from scripts import wqq_campaign_study as study

np, torch = base.np, base.torch
from scipy.ndimage import uniform_filter1d

DEFAULT_SOURCE = study.CAMPAIGN / "studies/layer1-v70-prior"
DEFAULT_STUDY = study.CAMPAIGN / "studies/layer4-v70-missing25"
ENDPOINTS = (0., .25, .5, .75)
Q, WINDOW = .98, 25
KIND = "wqq-campaign-missing25-study-v1"


def gate_spec():
    return {"mask": "not finite(actual raw ball coordinates).all(axis=1)",
            "window_frames": WINDOW, "window_mode": "centered-nearest-within-half",
            "alpha": "(1-missing25)*alpha_observed + missing25*alpha_missing",
            "alpha_dtype": "float32; constant policies use exact base.compose"}


def policy_grid():
    pairs = [(a, a) for a in (*ENDPOINTS, 1.)]
    pairs += [(a, b) for a in ENDPOINTS for b in ENDPOINTS if a != b]
    return [{"id": f"o{round(a * 100):03d}_m{round(b * 100):03d}",
             "alpha_observed": a, "alpha_missing": b, "parent_confidence": Q,
             "unchanged": a == b == 0., "complexity": int(a != b)} for a, b in pairs]


def validate_policy(policy):
    canonical = [{k: v for k, v in p.items() if k != "id"} for p in policy_grid()]
    if {k: v for k, v in policy.items() if k != "id"} not in canonical:
        raise ValueError("Policy is outside the frozen 17-entry missing25 grid")
    if not isinstance(policy.get("id"), str) or not policy["id"]:
        raise ValueError("A named registered endpoint policy is required")


def missing_mask(raw):
    if set(raw) != {"ball", "players", "team", "pitch"}:
        raise ValueError("Only the actual raw observation fields are accepted")
    ball = np.asarray(raw["ball"])
    if ball.ndim != 2 or ball.shape[1] != 2 or not len(ball):
        raise ValueError("A nonempty native-frame ball coordinate array is required")
    return ~np.isfinite(ball).all(1)


def missing_fraction(mask):
    mask = np.asarray(mask)
    if mask.ndim != 1 or not len(mask) or mask.dtype != np.bool_:
        raise ValueError("A nonempty boolean observation-derived missing mask is required")
    return uniform_filter1d(mask.astype(np.float64), WINDOW, mode="nearest")


def compose(parent, logits, observed_missing, policy):
    """No condition, match, reference, confidence teacher or mutable gate state."""
    validate_policy(policy)
    parent, logits = np.asarray(parent), np.asarray(logits, np.float32)
    fraction = missing_fraction(observed_missing)
    if (parent.ndim != 1 or parent.dtype.kind not in "iu" or not len(parent) or
            not np.isin(parent, [0, 1, 2, 3]).all() or len(fraction) != len(parent) or
            logits.shape != (len(parent), 5) or not np.isfinite(logits).all()):
        raise ValueError("Aligned finite native logits, canonical parent and raw mask required")
    a, b = policy["alpha_observed"], policy["alpha_missing"]
    if a == b:
        return base.compose(parent, logits, a, Q)
    alpha = ((1. - fraction) * a + fraction * b).astype(np.float32)[:, None]
    tensor = torch.from_numpy(logits)
    probability = base.emissions_api.emissions(tensor[:, 0], tensor[:, 1]).exp().numpy()
    prior = np.full_like(probability, (1. - Q) / 2.)
    known = parent < 3
    prior[np.flatnonzero(known), parent[known].astype(int)] = Q
    prior[~known] = 1. / 3.
    nodes = np.log(np.clip((1. - alpha) * prior + alpha * probability, 1e-8, 1.))
    return base.native.decode(nodes, logits[:, 2:])


def view_key(which, match, half, condition):
    study.parents._validate_key(which, match, half, condition)
    return f"{which}/{match}-{half}-{condition}"


def source_pins():
    sources = study.source_pins()
    for path in (Path(__file__), Path(runtime.__file__),
                 ROOT / "scripts/wqq_campaign_baseline_ancestry.py",
                 ROOT / "tests/test_wqq_campaign_gate.py"):
        sources[base.relative(path)] = base.sha256(path)
    return sources


def _cache_entry(source, source_protocol, which, match, half, condition):
    """Inspect existing receipts only; missing caches fail before any producer."""
    key = view_key(which, match, half, condition)
    if which not in base.TRAIN or match not in (which, "J03WPY"):
        raise PermissionError("The reuse index contains only outer and WPY views")
    name = key.split("/", 1)[1]
    view_path = Path(source) / "views" / which / f"{name}.json"
    logit_path = Path(source) / "logits" / which / f"{name}.json"
    parent_path = study.parents.CAMPAIGN / "parents-v2" / which / f"{name}.json"
    for path in (view_path, logit_path, parent_path):
        if not path.is_file():
            raise FileNotFoundError("Required frozen cache receipt is absent: " + str(path))
    view, logit, parent = [base.read_json(p) for p in (view_path, logit_path, parent_path)]
    identity = {"which": which, "match": match, "half": half, "condition": condition}
    if (any(view["identity"].get(k) != v or logit.get(k) != v or parent.get(k) != v
            for k, v in identity.items()) or view["identity"].get("protocol") != source_protocol or
            logit.get("family") != "v70" or logit.get("reference_reads") != [] or
            view.get("reference_reads") != [] or parent.get("nomination") is not None or
            logit.get("corruption_seed") != (100 if condition == "corrupted" else None)):
        raise ValueError("Source cache identity, observations or role differ")
    if base.pin(logit_path) not in view["inputs"] or base.pin(parent_path) not in view["inputs"]:
        raise ValueError("Source composed view does not bind these exact cached ancestors")
    if parent["source"] != base.pin(Path(study.parents.__file__)):
        raise ValueError("Parent cache producer source changed")
    if parent.get("ancestry_verified") is not True or not parent.get("upstream_evidence"):
        raise ValueError("Parent cache lacks verified upstream ancestry evidence")
    if which in parent["fit_matches"] or which in parent["calibration_matches"]:
        raise ValueError("Parent cache includes the outer held match")
    raw = base.pin(base.DATA / f"{match}-{half}.npz")
    if logit["raw_input"] != raw:
        raise ValueError("Logit raw fixture does not match the requested physical half")
    base.verify_sources(logit["sources"])
    references = {"source_view": base.pin(view_path), "source_output": view["artifact"],
                  "logit_receipt": base.pin(logit_path), "logits": logit["output"],
                  "parent_receipt": base.pin(parent_path), "parent": parent["artifact"], "raw_input": raw}
    for ref in (*references.values(), *parent["upstream_evidence"]):
        base.checked(ref)
    return {"identity": identity, **references}


def register(folder=DEFAULT_STUDY, source=DEFAULT_SOURCE):
    folder, source = Path(folder).resolve(), Path(source).resolve()
    folder.relative_to(study.CAMPAIGN / "studies")
    source.relative_to(study.CAMPAIGN / "studies")
    if folder == source:
        raise ValueError("The new study must preserve its source study")
    original = study.protocol_for(source)
    if original["recipe"]["model_family"] != "v70":
        raise ValueError("Only the frozen native-v70 source study may supply logits")
    source_protocol = base.pin(source / "protocol.json")
    seal, _ = study.evaluation.load_seal(source / "oof-seal.json", stage="oof")
    if seal["run_id"] != source.name or seal["protocol"] != source_protocol:
        raise ValueError("The existing source OOF seal has another identity")
    index = {}
    for which in base.TRAIN:
        for match in (which, "J03WPY"):
            for half in base.HALVES:
                for condition in ("clean", "corrupted"):
                    key = view_key(which, match, half, condition)
                    index[key] = _cache_entry(source, source_protocol, which, match, half, condition)
    recipe = {"model_family": "v70", "model_protocol": original["recipe"]["model_protocol"],
              "composition": "raw-missing25-native-three-state-v1", "policies": policy_grid(),
              "parent": original["recipe"]["parent"], "parent_confidence": Q,
              "gate": gate_spec(),
              "selection": "one equal-condition full-match WPY policy per producer; exact ties macroF1, unchanged, constants first, registry order",
              "transition": "unchanged v70 typed hazards, nonnegative costs and full-half decoding",
              "conditions": ["clean", "corrupted"], "corruption_seed": 100}
    protocol = {"kind": KIND, "candidate_id": folder.name, "recipe": recipe,
                "source_hashes": source_pins(), "source_study": base.relative(source),
                "source_protocol": source_protocol, "source_oof_seal": base.pin(source / "oof-seal.json"),
                "cache_index": index, "train_matches": list(base.TRAIN), "calibration_matches": ["J03WPY"],
                "target_match": "J03WQQ", "target_role": "one audit after exact eligible OOF admission",
                "controls": ["v70", "v69"], "primary_metric": "equal clean/corrupted official five-IoU full-match SpellF1",
                "cohort_selection": "physical OOF mean; no grid refinement after held audit",
                "no_human_challenge_annotations": True}
    base.atomic_json(folder / "protocol.json", protocol, immutable=True)
    return protocol


def protocol_for(folder):
    folder = Path(folder)
    protocol = study.protocol_for(folder)
    if (protocol["kind"] != KIND or protocol["recipe"]["policies"] != policy_grid() or
            protocol["recipe"]["model_family"] != "v70" or protocol["recipe"]["parent_confidence"] != Q or
            protocol["recipe"].get("gate") != gate_spec() or
            protocol["recipe"].get("composition") != "raw-missing25-native-three-state-v1" or
            protocol["recipe"].get("conditions") != ["clean", "corrupted"] or
            protocol["recipe"].get("corruption_seed") != 100):
        raise ValueError("Unregistered missing25 study protocol")
    base.checked(protocol["source_protocol"])
    base.checked(protocol["source_oof_seal"])
    # Full inference must use the same feature/native implementation as the
    # pinned model protocol; do not parse or hash its training-reference inputs.
    model_protocol = base.read_json(base.checked(protocol["recipe"]["model_protocol"]))
    base.verify_sources(model_protocol["sources"])
    return protocol


def _inputs(folder, protocol, which, match, half, condition, permit):
    key = view_key(which, match, half, condition)
    base.validate_inference_role(which, match, permit, "v70")
    if which == "full":
        if permit is None:
            raise PermissionError("Every full-model gate view requires exact OOF admission")
        runtime.preflight_full(folder, permit)
        path = Path(folder) / "logits" / which / f"{key.split('/', 1)[1]}.npz"
        logit = base.predict_logits("v70", which, match, half, condition, path, permit=permit)
        parent_values, parent = study.parents.get_parent(which, match, half, condition, nomination=permit)
        refs = {"logit_receipt": base.pin(path.with_suffix(".json")), "logits": logit["output"],
                "parent_receipt": parent["manifest"], "parent": parent["artifact"], "raw_input": logit["raw_input"]}
    else:
        if key not in protocol["cache_index"]:
            raise PermissionError("View absent from the prospectively frozen cache index")
        entry = protocol["cache_index"][key]
        actual = _cache_entry(ROOT / protocol["source_study"], protocol["source_protocol"], which, match, half, condition)
        if entry != actual:
            raise ValueError("Frozen input identity or cached ancestor changed")
        refs = {k: v for k, v in entry.items() if k != "identity"}
        logit = base.read_json(base.checked(refs["logit_receipt"]))
        parent = base.read_json(base.checked(refs["parent_receipt"]))
        parent["manifest"] = refs["parent_receipt"]
        with np.load(base.checked(refs["parent"]), allow_pickle=False) as data:
            parent_values = data["parent"].copy()
    with np.load(base.checked(refs["logits"]), allow_pickle=False) as data:
        logits = data["logits"].copy()
    if logits.shape != (len(parent_values), 5) or not np.isfinite(logits).all():
        raise ValueError("Cache arrays are not finite and frame aligned")
    return parent_values, logits, logit, parent, refs


def _mask(folder, identity, raw_pin, frames):
    """A mask belongs to one explicit view identity, never ambient adapter state."""
    path = Path(folder) / "masks" / identity["which"] / f"{identity['match']}-{identity['half']}-{identity['condition']}.npz"
    receipt = path.with_suffix(".json")
    mask_identity = {**identity, "raw_input": raw_pin, "frames": frames,
                     "corruption_seed": 100 if identity["condition"] == "corrupted" else None}
    if receipt.exists():
        meta = base.read_json(receipt)
        if meta["identity"] != mask_identity or meta.get("reference_reads") != []:
            raise ValueError("Missing-mask resume identity mismatch")
        base.checked(raw_pin)
        with np.load(base.checked(meta["artifact"]), allow_pickle=False) as data:
            mask, fraction = data["missing"].copy(), data["missing25"].copy()
        if len(mask) != frames or not np.array_equal(missing_fraction(mask), fraction):
            raise ValueError("Missing-mask cache has inconsistent frame alignment or smoothing")
        return mask, meta, base.pin(receipt)
    if path.exists():
        raise FileExistsError("Unsealed missing mask requires inspection")
    raw, actual_pin = base.raw_fixture(identity["match"], identity["half"], identity["condition"])
    if actual_pin != raw_pin:
        raise ValueError("Gate observations do not match the native logit input")
    mask = missing_mask(raw)
    if len(mask) != frames:
        raise ValueError("Raw gate observations and cached logits have different frame counts")
    ball = np.ascontiguousarray(raw["ball"])
    ball_hash = hashlib.sha256(str((ball.dtype.str, ball.shape)).encode() + ball.tobytes()).hexdigest()
    base.atomic_npz(path, missing=mask, missing25=missing_fraction(mask))
    meta = {"kind": "wqq-campaign-observation-mask-v1", "identity": mask_identity,
            "artifact": base.pin(path), "actual_ball_sha256": ball_hash, "reference_reads": []}
    base.atomic_json(receipt, meta, immutable=True)
    return mask, meta, base.pin(receipt)


def produce_view(folder, which, match, half, condition, policies, permit=None):
    folder = Path(folder)
    protocol = protocol_for(folder)
    for policy in policies:
        validate_policy(policy)
    parent, logits, logit_meta, parent_meta, refs = _inputs(folder, protocol, which, match, half, condition, permit)
    identity = {"which": which, "match": match, "half": half, "condition": condition,
                "protocol": base.pin(folder / "protocol.json"), "input_identity": refs}
    mask, mask_meta, mask_receipt = _mask(folder, identity, refs["raw_input"], len(parent))
    identity = {**identity, "policies": policies, "mask": mask_meta["artifact"], "mask_receipt": mask_receipt}
    destination = folder / "views" / which / f"{match}-{half}-{condition}.npz"
    receipt = destination.with_suffix(".json")
    if receipt.exists():
        meta = base.read_json(receipt)
        if meta["identity"] != identity:
            raise ValueError("Gate composition resume identity mismatch")
        for ref in [meta["artifact"], *meta["inputs"]]:
            base.checked(ref)
        return meta
    if destination.exists():
        raise FileExistsError("Unsealed gate composition requires inspection")
    arrays = {p["id"]: compose(parent, logits, mask, p) for p in policies}
    base.atomic_npz(destination, **arrays)
    nodes = {}
    native_roots = [study.weight_nodes(a["freeze"], nodes) for a in logit_meta["model_bundle"]["learned_ancestors"]]
    parent_name = "parent-" + parent_meta["manifest"]["sha256"]
    nodes[parent_name] = {"kind": "historical_parent", "fit_matches": parent_meta["fit_matches"],
                          "calibration_matches": parent_meta["calibration_matches"], "dependencies": [],
                          "artifacts": [parent_meta["manifest"], parent_meta["artifact"], parent_meta["source"]],
                          "role_receipt": parent_meta["manifest"]}
    # Numerically identical logits can arise from distinct physical inputs;
    # their receipts and raw-input dependencies still need separate nodes.
    native_name = "logits-" + refs["logits"]["sha256"] + "-" + view_key(which, match, half, condition)
    nodes[native_name] = {"kind": "pipeline", "fit_matches": [], "calibration_matches": [],
                          "dependencies": native_roots,
                          "artifacts": [refs["logit_receipt"], refs["logits"], refs["raw_input"],
                                        *[{"path": p, "sha256": s} for p, s in logit_meta["sources"].items()]]}
    output_name = "gate-" + base.pin(destination)["sha256"] + "-" + view_key(which, match, half, condition)
    inputs = [*refs.values(), mask_receipt, mask_meta["artifact"], base.pin(folder / "protocol.json")]
    nodes[output_name] = {"kind": "pipeline", "fit_matches": [], "calibration_matches": [],
                          "dependencies": [parent_name, native_name], "artifacts": [base.pin(destination), *inputs]}
    meta = {"kind": "wqq-campaign-composition-v1", "identity": identity, "artifact": base.pin(destination),
            "inputs": inputs, "ancestry_nodes": nodes, "ancestry_roots": [output_name],
            "reference_reads": [], "arrays": {p["id"]: p["id"] for p in policies}, "frames": len(parent)}
    base.atomic_json(receipt, meta, immutable=True)
    study.emit(stage="prediction", study=folder.name, which=which, match=match, half=half,
               condition=condition, frames=len(parent), policies=len(policies), cached_native=which != "full")
    study.checkpoint(folder, base.relative(receipt))
    return meta


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
    admission["gate_adapter"] = base.pin(Path(__file__))
    base.atomic_json(Path(folder) / "full-admission.json", admission, immutable=True)
    with installed():
        return study.full_audit(Path(folder), permit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "outer", "oof-audit", "full"))
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--source-study", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--held", choices=base.TRAIN)
    parser.add_argument("--permit", type=Path)
    args = parser.parse_args()
    folder = args.study.resolve()
    folder.relative_to(study.CAMPAIGN / "studies")
    base.threadpool_limits(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    sys.addaudithook(study.guard)
    try:
        if args.command == "register":
            register(folder, args.source_study)
            study.emit(stage="registered", study=folder.name, policies=17)
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
