"""Sealed, condition-balanced evaluation for the WQQ 0.90 research campaign.

The scientific inference function never receives a benchmark condition. Conditions
here identify separately prepared observations and receive equal evaluation weight.
This module does not fit models, produce predictions, or open reserved references
until every required prediction has been loaded and its provenance checked.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

from scripts.full_baseline_evaluate import changes, join_halves, score_full
from second_zero.evaluate import official_metrics

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "wqq-090-20261031"
TRAIN = ("J03WMX", "J03WN1", "J03WOH", "J03WOY")
HALVES = ("firstHalf", "secondHalf")
CONDITIONS = ("clean", "corrupted")
STAGE_MATCHES = {"calibration": ("J03WPY",), "oof": TRAIN, "full": ("J03WQQ",)}
CONTROL_FAMILIES = {"v70": "hierarchical-joint-v70", "v69": "capped-fusion-v69"}
SCHEMA = "wqq-campaign-prediction-seal-v1"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def recipe_sha256(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path(path, root=ROOT, *, full_authorized=False, reference=False):
    root = Path(root).resolve()
    path = Path(path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact escapes the declared workspace")
    name = str(path)
    if "J03WR9" in name or "/evaluation/gt/" in name or "/runs/main/" in name:
        raise PermissionError("Reserved WR9 and challenge references remain closed")
    if "J03WQQ" in name and not full_authorized:
        raise PermissionError("WQQ is closed until a frozen nomination")
    if (".truth." in path.name or ".actor." in path.name) and not reference:
        raise PermissionError("Label payload is not a prediction/provenance artifact")
    return path


def pin(path, root=ROOT, *, full_authorized=False):
    path = _path(path, root, full_authorized=full_authorized)
    return {"path": str(path.relative_to(Path(root).resolve())), "sha256": sha256(path)}


def checked(ref, root=ROOT, *, full_authorized=False):
    if not isinstance(ref, dict) or set(("path", "sha256")) - ref.keys():
        raise ValueError("A hash-bound artifact reference is required")
    p = _path(ref["path"], root, full_authorized=full_authorized)
    if sha256(p) != ref["sha256"]:
        raise ValueError("Changed immutable artifact: " + ref["path"])
    return p


def read(path):
    with Path(path).open() as f:
        return json.load(f)


def write_new(path, data):
    """Immutable JSON receipt, atomically published without replacing a prior run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    try:
        with temporary.open("x") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def labels(value):
    x = np.asarray(value)
    if x.ndim != 1 or not len(x) or x.dtype.kind not in "iu" or not np.isin(x, [0, 1, 2, 3]).all():
        raise ValueError("Nonempty one-dimensional integer H/A/D/X labels required")
    return x.astype(np.int8, copy=True)


def _dotted(value, field):
    for key in field.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def validate_ancestry(nodes, roots, held_match, *, root=ROOT, full_authorized=False):
    """Walk every functional ancestor and bind fitted roles to original receipts.

    A node is {kind, dependencies, fit_matches, calibration_matches, artifacts,
    role_receipt}. Fitted nodes need a role_receipt pin; its optional fit_field and
    calibration_field select the original receipt fields. Raw/pipeline nodes have
    no labels and must enumerate their dependencies explicitly.
    """
    if not isinstance(nodes, dict) or not roots:
        raise ValueError("Nonempty functional ancestry roots are required")
    seen, visiting = set(), set()
    allowed_fit = set(TRAIN) - {held_match}

    def visit(name):
        if name in visiting:
            raise ValueError("Cyclic functional ancestry")
        if name in seen:
            return
        if name not in nodes:
            raise ValueError("Missing functional ancestor: " + name)
        n = nodes[name]
        visiting.add(name)
        if "dependencies" not in n:
            raise ValueError("Every ancestor must explicitly enumerate dependencies")
        fit = set(n.get("fit_matches", []))
        cal = set(n.get("calibration_matches", []))
        kind = n.get("kind")
        allowed_cal = allowed_fit | {"J03WPY"} if kind == "historical_parent" else {"J03WPY"}
        if not fit <= allowed_fit or not cal <= allowed_cal:
            raise PermissionError("Functional ancestor crosses declared label roles: " + name)
        if kind not in {"weights", "calibration", "historical_parent", "raw", "features", "pipeline"}:
            raise ValueError("Unknown ancestor kind")
        if kind in {"weights", "historical_parent"} and not fit:
            raise ValueError("Fitted weights require an explicit native fitting cohort")
        if kind == "calibration" and cal != {"J03WPY"}:
            raise ValueError("A calibrated node must declare WPY")
        if kind in {"raw", "features", "pipeline"} and (fit or cal):
            raise ValueError("Label-dependent nodes must declare a fitted kind")
        artifacts = n.get("artifacts", [])
        if kind in {"weights", "calibration", "historical_parent"} and not artifacts:
            raise ValueError("Fitted nodes must bind their artifacts")
        for artifact in artifacts:
            checked(artifact, root, full_authorized=full_authorized)
        if fit or cal:
            ref = n.get("role_receipt")
            original = read(checked(ref, root, full_authorized=full_authorized))
            if fit and set(_dotted(original, ref.get("fit_field", "fit_matches"))) != fit:
                raise ValueError("Weight cohort disagrees with original receipt")
            if cal and set(_dotted(original, ref.get("calibration_field", "calibration_matches"))) != cal:
                raise ValueError("Calibration cohort disagrees with original receipt")
            if kind == "historical_parent":
                if original.get("ancestry_verified") is not True or not original.get("upstream_evidence"):
                    raise ValueError("Historical parent requires verified upstream ancestry evidence")
                for evidence in original["upstream_evidence"]:
                    checked(evidence, root, full_authorized=full_authorized)
            # Optional source-receipt artifact bindings prevent accidental pairing
            # of a role receipt with a different model's otherwise valid hash.
            for binding in n.get("artifact_bindings", []):
                if _dotted(original, binding["field"]) != binding["sha256"]:
                    raise ValueError("Artifact disagrees with original role receipt")
                if binding["sha256"] not in {a["sha256"] for a in artifacts}:
                    raise ValueError("Role binding is not one of this node's artifacts")
        for dependency in n["dependencies"]:
            visit(dependency)
        visiting.remove(name)
        seen.add(name)

    for name in roots:
        visit(name)
    return sorted(seen)


def _validate_policies(policies):
    if not policies or any(not isinstance(p.get("id"), str) or not p["id"] for p in policies):
        raise ValueError("Policies require stable nonempty string ids")
    ids = [p["id"] for p in policies]
    if len(ids) != len(set(ids)):
        raise ValueError("Policy ids must be unique")
    return ids


def _nomination(ref, run_id, protocol, root):
    if ref is None:
        raise PermissionError("Full WQQ work requires a frozen nomination")
    d = read(checked(ref, root))
    if d.get("kind") != "wqq-campaign-nomination-v1" or d.get("candidate_id") != run_id:
        raise PermissionError("Run does not match the frozen nomination")
    if d.get("selected_run_id", run_id) != run_id:
        raise PermissionError("Conflicting nomination candidate aliases")
    if d.get("protocol") != protocol or not d.get("eligible"):
        raise PermissionError("Nomination is not bound to this eligible recipe")
    if recipe_sha256(d["selected_recipe"]) != d["recipe_sha256"]:
        raise PermissionError("Nomination recipe digest mismatch")
    if d.get("selected_oof_audit", d["oof_audit"]) != d["oof_audit"]:
        raise PermissionError("Conflicting nomination audit aliases")
    audit = read(checked(d["oof_audit"], root))
    if audit.get("stage") != "oof" or not audit.get("eligible") or audit.get("run_id") != run_id or audit.get("protocol") != protocol:
        raise PermissionError("Nomination lacks its matching successful OOF audit")
    return d


def _load_rows(rows, policies, stage, nodes, *, root=ROOT, full_authorized=False, producer_held=None):
    ids = set(_validate_policies(policies))
    matches = STAGE_MATCHES[stage]
    if stage == "calibration" and producer_held not in (*TRAIN, "full"):
        raise ValueError("Calibration must declare its excluded producer match or full")
    expected = {(m, h, c) for m in matches for h in HALVES for c in CONDITIONS}
    cases, summaries, selections = {}, [], {}
    for row in rows:
        key = row["match"], row["half"], row["condition"]
        if key not in expected or key in cases:
            raise ValueError("Unexpected or duplicated physical match/half/condition")
        if set(row["arrays"]) != ids or len(set(row["arrays"].values())) != len(ids):
            raise ValueError("Every view must contain exactly the registered policies")
        excluded = producer_held if stage == "calibration" and producer_held != "full" else key[0]
        ancestors = validate_ancestry(nodes, row["ancestry_roots"], excluded, root=root, full_authorized=full_authorized)
        selection_ref = None
        if stage != "calibration":
            selection_ref = row.get("policy_selection")
            selection = read(checked(selection_ref, root))
            if (selection.get("kind") != "wqq-campaign-calibration-v1"
                    or selection.get("calibration_matches") != ["J03WPY"]
                    or selection.get("policy_shared_across_conditions") is not True
                    or selection.get("producer_held") != (key[0] if stage == "oof" else "full")
                    or not isinstance(selection.get("selected"), dict)
                    or any(r.get("match") != "J03WPY" for r in selection.get("reference_reads", []))):
                raise ValueError("Candidate requires one verified WPY-only policy selection")
            checked(selection["prediction_seal"], root)
            if key[0] in selections and selections[key[0]] != selection_ref:
                raise ValueError("A physical match must share one policy across both halves and conditions")
            selections[key[0]] = selection_ref
        ref = row.get("artifact") or pin(row["path"], root, full_authorized=full_authorized)
        path = checked(ref, root, full_authorized=full_authorized)
        with np.load(path, allow_pickle=False) as z:
            arr = {name: labels(z[array]) for name, array in row["arrays"].items()}
        checked(ref, root, full_authorized=full_authorized)
        if len({len(a) for a in arr.values()}) != 1:
            raise ValueError("Policies have different frame coverage")
        cases[key] = arr
        summaries.append({"match": key[0], "half": key[1], "condition": key[2], "artifact": ref,
                          "arrays": row["arrays"], "frames": len(next(iter(arr.values()))),
                          "ancestry_roots": row["ancestry_roots"], "ancestors_verified": ancestors,
                          "policy_selection": selection_ref})
    if set(cases) != expected:
        raise ValueError("Incomplete physical-match/half/condition coverage")
    for m in matches:
        for h in HALVES:
            lengths = {len(next(iter(cases[m, h, c].values()))) for c in CONDITIONS}
            if len(lengths) != 1:
                raise ValueError("Clean/corrupted source-frame coverage differs")
    return cases, summaries


def create_seal(path, *, run_id, stage, rows, policies, protocol, ancestry_nodes,
                root=ROOT, nomination=None, producer_held=None):
    if stage not in STAGE_MATCHES:
        raise ValueError("Unknown campaign evaluation stage")
    checked(protocol, root)
    full = stage == "full"
    if full:
        _nomination(nomination, run_id, protocol, root)
    _, verified = _load_rows(rows, policies, stage, ancestry_nodes, root=root, full_authorized=full, producer_held=producer_held)
    result = {"kind": SCHEMA, "campaign": CAMPAIGN, "run_id": run_id, "created_utc": now(),
              "stage": stage, "producer_held": producer_held, "protocol": protocol, "policies": policies, "rows": verified,
              "ancestry_nodes": ancestry_nodes, "reference_reads": [], "nomination": nomination,
              "condition_use": "Benchmark input construction only; one inference recipe and policy across conditions"}
    path = _path(path, root, full_authorized=full)
    write_new(path, result)
    return result


def load_seal(path, *, root=ROOT, stage=None):
    # The seal's name need not include WQQ; authorization is checked before any
    # WQQ prediction or source artifact is opened.
    p = _path(path, root)
    d = read(p)
    if d.get("kind") != SCHEMA or d.get("campaign") != CAMPAIGN or d.get("reference_reads") != []:
        raise ValueError("Invalid or previously reference-dependent prediction seal")
    if d.get("stage") not in STAGE_MATCHES or (stage and d["stage"] != stage):
        raise ValueError("Wrong evaluation stage")
    checked(d["protocol"], root)
    full = d["stage"] == "full"
    if full:
        _nomination(d.get("nomination"), d["run_id"], d["protocol"], root)
    cases, summaries = _load_rows(d["rows"], d["policies"], d["stage"], d["ancestry_nodes"], root=root, full_authorized=full,
                                 producer_held=d.get("producer_held"))
    if summaries != d["rows"]:
        raise ValueError("Sealed prediction metadata changed")
    return d, cases


def select_policy(cases, truth_by_half, policies):
    """Select one policy for both conditions from sealed WPY prediction arrays.

    Pure numerical interface: cases[(half, condition)][policy_id] -> labels;
    truth_by_half[half] -> labels. File callers must use select_policy_from_seal.
    """
    ids = _validate_policies(policies)
    if set(cases) != {(h, c) for h in HALVES for c in CONDITIONS} or set(truth_by_half) != set(HALVES):
        raise ValueError("Both complete halves and both conditions are required")
    y = join_halves([labels(truth_by_half[h]) for h in HALVES])
    pm = official_metrics()
    rows = []
    for index, policy in enumerate(policies):
        pid = policy["id"]
        details = {}
        for c in CONDITIONS:
            if any(set(cases[h, c]) != set(ids) for h in HALVES):
                raise ValueError("Per-condition policies differ")
            parts = [labels(cases[h, c][pid]) for h in HALVES]
            if any(len(p) != len(truth_by_half[h]) for p, h in zip(parts, HALVES)):
                raise ValueError("Prediction/reference source-frame mismatch")
            details[c] = score_full(pm, y, join_halves(parts))[0]
        rows.append({"id": pid, "policy": policy, "index": index,
                     "spell_f1_mean": float(np.mean([details[c]["spell_f1_mean"] for c in CONDITIONS])),
                     "macro_f1": float(np.mean([details[c]["macro_f1"] for c in CONDITIONS])),
                     "conditions": details})
    def rank(row):
        p = row["policy"]
        unchanged = bool(p.get("unchanged", p.get("alpha") == 0))
        return (row["spell_f1_mean"], row["macro_f1"], unchanged,
                -float(p.get("complexity", row["index"])), -row["index"])
    winner = max(rows, key=rank)
    return {"selected": winner["policy"], "selected_id": winner["id"], "selected_score": winner["spell_f1_mean"],
            "rows": rows, "policy_shared_across_conditions": True,
            "selection": "Equal clean/corrupted official full-match SpellF1; exact ties macroF1, unchanged, simplicity, registry order"}


def _truths(matches, cases, *, root=ROOT, truth_loader=None, full_authorized=False):
    result, reads = {}, []
    for m in matches:
        for h in HALVES:
            if m == "J03WQQ" and not full_authorized:
                raise PermissionError("WQQ truth is not authorized")
            if truth_loader is None:
                p = _path(Path("private/improvement/external/prepared") / f"{m}-{h}.truth.npy", root,
                          full_authorized=full_authorized, reference=True)
                y = labels(np.load(p, allow_pickle=False))
                evidence = {"path": str(p.relative_to(Path(root).resolve())), "sha256": sha256(p)}
            else:
                y = labels(truth_loader(m, h))
                evidence = {"loader": "explicit injected reference loader"}
            if any(len(next(iter(cases[m, h, c].values()))) != len(y) for c in CONDITIONS):
                raise ValueError("Reference/prediction source-frame mismatch")
            result[m, h] = y
            reads.append({"match": m, "half": h, **evidence})
    return result, reads


def select_policy_from_seal(path, *, root=ROOT, truth_loader=None, out=None):
    d, cases = load_seal(path, root=root, stage="calibration")
    truth, reads = _truths(("J03WPY",), cases, root=root, truth_loader=truth_loader)
    result = select_policy({(h, c): cases["J03WPY", h, c] for h in HALVES for c in CONDITIONS},
                           {h: truth["J03WPY", h] for h in HALVES}, d["policies"])
    result.update(kind="wqq-campaign-calibration-v1", run_id=d["run_id"], protocol=d["protocol"],
                  prediction_seal=pin(path, root), calibration_matches=["J03WPY"], reference_reads=reads,
                  producer_held=d["producer_held"], created_utc=now())
    if out:
        write_new(_path(out, root), result)
    return result


def _controls(stage, matches, *, root=ROOT, control_paths=None):
    result, receipts = {}, {}
    directory = "full-predictions" if stage == "full" else "oof"
    for name, family in CONTROL_FAMILIES.items():
        sp = (control_paths or {}).get(name, Path("runs") / family / ("full-seal.json" if stage == "full" else "oof-seal.json"))
        sp = _path(sp, root, full_authorized=stage == "full")
        seal = read(sp)
        if seal.get("reference_reads") != []:
            raise ValueError("Control predictions were not sealed before references")
        arrays = {}
        for m in matches:
            for h in HALVES:
                for c in CONDITIONS:
                    suffix = f"/{directory}/{m}-{h}-{c}.npz"
                    paths = [(p, sha) for p, sha in seal["outputs"].items() if p.endswith(suffix)]
                    if len(paths) != 1:
                        raise ValueError("Control seal lacks unique required view")
                    p, sha = paths[0]
                    ref = {"path": p, "sha256": sha}
                    with np.load(checked(ref, root, full_authorized=stage == "full"), allow_pickle=False) as z:
                        arrays[m, h, c] = labels(z["joint"])
                    checked(ref, root, full_authorized=stage == "full")
        result[name] = arrays
        receipts[name] = pin(sp, root, full_authorized=stage == "full")
    return result, receipts


def _audit_predictions(path, *, root=ROOT, truth_loader=None, out=None, control_paths=None):
    d, cases = load_seal(path, root=root)
    stage = d["stage"]
    if stage not in {"oof", "full"} or len(d["policies"]) != 1:
        raise ValueError("Audit requires one already-selected candidate, not a policy grid")
    pid = d["policies"][0]["id"]
    matches = STAGE_MATCHES[stage]
    controls, control_receipts = _controls(stage, matches, root=root, control_paths=control_paths)
    for key, row in cases.items():
        if any(len(c[key]) != len(row[pid]) for c in controls.values()):
            raise ValueError("Control/candidate frame coverage differs")
    claim_path = None
    if stage == "full":
        if out is None:
            raise ValueError("Full audit requires an immutable output path")
        claim_path = checked(d["nomination"], root).with_suffix(".full-audit-claim.json")
        claim = {"kind": "wqq-campaign-full-audit-claim-v1", "nomination": d["nomination"],
                 "prediction_seal": pin(path, root), "output": str(_path(out, root).relative_to(Path(root).resolve()))}
        if claim_path.exists():
            if read(claim_path) != claim:
                raise PermissionError("Nomination was already consumed by a different frozen full audit")
            completed = _path(claim["output"], root)
            if completed.exists():
                result = read(completed)
                if result.get("prediction_seal") != claim["prediction_seal"] or result.get("nomination") != d["nomination"]:
                    raise ValueError("Existing full audit is not bound to its claim")
                return result
            # A crashed deterministic evaluation may resume only the exact same
            # frozen prediction seal/output. It never selects another candidate.
        else:
            write_new(claim_path, claim)
    # All candidate and control arrays are validated in memory before any truth.
    truth, reads = _truths(matches, cases, root=root, truth_loader=truth_loader, full_authorized=stage == "full")
    pm = official_metrics()
    names = ("v70", "v69", "candidate")
    details, differences = {}, {}
    for m in matches:
        y = join_halves([truth[m, h] for h in HALVES])
        details[m], differences[m] = {}, {}
        for c in CONDITIONS:
            pred = {name: join_halves([controls[name][m, h, c] for h in HALVES]) for name in controls}
            pred["candidate"] = join_halves([cases[m, h, c][pid] for h in HALVES])
            metrics, internals = {}, {}
            for name in names:
                metrics[name], internals[name] = score_full(pm, y, pred[name])
            details[m][c] = metrics
            differences[m][c] = {name: changes(internals[name], internals["candidate"]) for name in controls}
    def average(ms, cs, metric="spell_f1_mean"):
        return {name: float(np.mean([details[m][c][name][metric] for m in ms for c in cs])) for name in names}
    overall = average(matches, CONDITIONS)
    pairs = {} if stage == "full" else {str(i): average(TRAIN[i*2:i*2+2], CONDITIONS) for i in (0, 1)}
    eligible = overall["candidate"] > overall["v70"]
    result = {"kind": "wqq-campaign-audit-v1", "created_utc": now(), "run_id": d["run_id"], "stage": stage,
              "protocol": d["protocol"], "prediction_seal": pin(path, root), "control_seals": control_receipts,
              "overall": overall, "macro_f1": average(matches, CONDITIONS, "macro_f1"),
              "conditions": {c: average(matches, (c,)) for c in CONDITIONS},
              "per_match": {m: average((m,), CONDITIONS) for m in matches}, "per_producer_pair": pairs,
              "details": details, "gained_lost_vs_controls": differences, "reference_reads": reads,
              "eligible": eligible, "delta_vs_v70": overall["candidate"]-overall["v70"],
              "clean_identity_required": False, "WQQ_fitted_or_calibrated": False,
              "nomination": d.get("nomination"),
              "gate": "Strict paired mean gain over v70; per-condition/pair regressions reported, not hidden vetoes",
              "qualification": "Official both-half five-IoU SpellF1, equal conditions and physical-match means; adaptively reused development, not private competition accuracy"}
    # Official per-IoU dictionaries use float keys in memory. Normalize once so
    # fresh and resumed audit calls return exactly the same wire representation.
    result = json.loads(json.dumps(result, allow_nan=False))
    if out:
        write_new(_path(out, root), result)
    return result


def audit_predictions(path, *, root=ROOT, truth_loader=None, out=None, control_paths=None):
    d = read(_path(path, root))
    kwargs = dict(root=root, truth_loader=truth_loader, out=out, control_paths=control_paths)
    if d.get("stage") != "full":
        return _audit_predictions(path, **kwargs)
    _nomination(d.get("nomination"), d["run_id"], d["protocol"], root)
    lease = checked(d["nomination"], root).with_suffix(".full-audit.lock")
    with lease.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This nomination already has a running full audit") from exc
        try:
            return _audit_predictions(path, **kwargs)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def nominate(audit_paths, out, *, root=ROOT, recipe=None):
    entries = []
    for path in audit_paths:
        p = _path(path, root)
        a = read(p)
        if a.get("kind") != "wqq-campaign-audit-v1" or a.get("stage") != "oof":
            raise ValueError("Nomination requires completed physical OOF audits")
        checked(a["protocol"], root)
        checked(a["prediction_seal"], root)
        if a["eligible"]:
            entries.append((a, pin(p, root)))
    if not entries:
        raise ValueError("No eligible candidate; preserve the existing best")
    winner, ref = max(entries, key=lambda pair: (pair[0]["overall"]["candidate"], pair[0]["macro_f1"]["candidate"]))
    if recipe is None:
        recipe = read(checked(winner["protocol"], root)).get("recipe", {"run_id": winner["run_id"], "protocol_sha256": winner["protocol"]["sha256"]})
    result = {"kind": "wqq-campaign-nomination-v1", "created_utc": now(), "eligible": True,
              "candidate_id": winner["run_id"], "selected_run_id": winner["run_id"], "oof_audit": ref, "selected_oof_audit": ref,
              "protocol": winner["protocol"], "candidate_audits": [r for _, r in entries],
              "selected_recipe": recipe, "recipe_sha256": recipe_sha256(recipe),
              "WQQ_reference_reads": [], "selection": "Highest eligible physical OOF SpellF1, macroF1 exact tie, input order final tie"}
    write_new(_path(out, root), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("seal", "select-policy", "audit", "nominate"))
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "seal":
        if len(args.inputs) != 1:
            p.error("seal accepts one inventory JSON")
        inventory = read(_path(args.inputs[0]))
        result = create_seal(args.out, **inventory)
    elif args.command == "select-policy":
        result = select_policy_from_seal(args.inputs[0], out=args.out)
    elif args.command == "audit":
        result = audit_predictions(args.inputs[0], out=args.out)
    else:
        result = nominate(args.inputs, args.out)
    print(json.dumps({k: result[k] for k in ("kind", "run_id", "selected_id", "selected_run_id", "overall", "eligible") if k in result}))


if __name__ == "__main__":
    main()
