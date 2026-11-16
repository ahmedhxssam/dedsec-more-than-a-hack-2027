"""Synthetic raw replays only: no historical prediction, model or truth loads."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts import wqq_campaign_transition_replay_v1 as replay


def raw(n=200):
    rng = np.random.default_rng(17)
    return {"ball": rng.normal(size=(n, 2)).astype(np.float32),
            "players": rng.normal(size=(n, 22, 2)).astype(np.float32),
            "team": np.repeat(np.array([0, 1], np.int8), 11),
            "pitch": np.array([105, 68], np.float32)}


def relocate(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(replay.chain, "ROOT", tmp_path)


def json_replace(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


def arrays(n=200):
    parent = np.resize(np.array([0, 1, 2], np.int8), n)
    logits = np.random.default_rng(30).normal(size=(n, 5)).astype(np.float32)
    return {"parent": parent, "logits": logits,
            "transition": replay.scale.compose(parent, logits, replay.POLICY)}


def test_pure_api_matches_frozen_composition_with_no_clean_bypass():
    a = arrays(); r = raw()
    bundle = NS(parent_from_raw=lambda v: (a["parent"].copy(), {"baseline": a["parent"].copy()}, {}),
                native_from_raw=lambda v: a["logits"].copy())
    before = replay.chain.raw_identity(r)
    actual, _ = replay.predict_raw(r, bundle)
    for k in a:
        assert replay.chain.array_pin(actual[k]) == replay.chain.array_pin(a[k])
    assert replay.chain.raw_identity(r) == before
    assert list(inspect.signature(replay.predict_raw).parameters) == ["raw", "bundle"]
    with pytest.raises(TypeError):
        replay.predict_raw(r, bundle, condition="clean")
    for field in ("truth", "condition", "match", "cached_parent"):
        with pytest.raises(ValueError):
            replay.predict_raw({**r, field: 1}, bundle)
    def mutated(v):
        v["ball"][0, 0] += 1
        return a["logits"]
    bundle.native_from_raw = mutated
    with pytest.raises(ValueError, match="mutated"):
        replay.predict_raw(r, bundle)


def test_registered_policy_is_exactly_the_existing_selected_entry():
    assert replay.POLICY in replay.scale.policy_grid()
    assert replay.POLICY["id"] == "s050_joint_a050"
    assert replay.POLICY["transition_scale"] == replay.POLICY["alpha"] == .5
    assert replay.POLICY["parent_confidence"] == .98


def candidate_tree(monkeypatch):
    """Minimal lifecycle records; deliberately no metric values or truth data."""
    names = ("protocol.json", "transition-scale-v1.json", "oof-audit.json", "full-admission.json",
             "full-audit.json", "full-seal.json", "selection.json", "parent-cache-admission-v1.json",
             "parent-cache-repair-v1.json", "independent.json")
    refs = {n: {"path": n, "sha256": n} for n in names}
    pp, nom = refs["protocol.json"], refs["transition-scale-v1.json"]
    oofseal = {"path": "oof-seal.json", "sha256": "oof"}
    calref = {"path": "cal-seal.json", "sha256": "cal"}
    recipe = {"model_family": "v70", "policies": replay.scale.policy_grid(),
              "model_protocol": {"path": "model-protocol.json", "sha256": "model"}}
    db = {
        replay.CANDIDATE["path"]: {"kind": "unpromoted_research_candidate", "study": replay.STUDY_NAME,
            "family": "hierarchical-joint-v70-transition-scale", "one_frozen_nomination": True,
            "prediction_sealed_before_reference_audit": True, "WQQ_fitted_or_calibrated": False,
            "independent_score_verification": True, "artifacts": list(refs.values())},
        "protocol.json": {"kind": replay.scale.KIND, "candidate_id": replay.STUDY_NAME,
            "recipe": recipe, "source_hashes": {}},
        "model-protocol.json": {"sources": {}},
        "transition-scale-v1.json": {"kind": "wqq-campaign-nomination-v1", "eligible": True,
            "candidate_id": replay.STUDY_NAME, "selected_run_id": replay.STUDY_NAME,
            "protocol": pp, "oof_audit": refs["oof-audit.json"], "selected_recipe": copy.deepcopy(recipe),
            "recipe_sha256": replay.base.canonical_recipe_hash(recipe), "WQQ_reference_reads": []},
        "oof-audit.json": {"kind": "wqq-campaign-audit-v1", "stage": "oof", "eligible": True,
            "run_id": replay.STUDY_NAME, "protocol": pp, "prediction_seal": oofseal},
        "full-admission.json": {"kind": "wqq-campaign-admission-v1", "nomination": nom,
            "protocol": pp, "oof_seal": oofseal, "reference_reads": []},
        "full-audit.json": {"kind": "wqq-campaign-audit-v1", "stage": "full", "run_id": replay.STUDY_NAME,
            "protocol": pp, "nomination": nom, "prediction_seal": refs["full-seal.json"],
            "WQQ_fitted_or_calibrated": False},
        "full-seal.json": {"kind": "wqq-campaign-prediction-seal-v1", "stage": "full",
            "run_id": replay.STUDY_NAME, "protocol": pp, "nomination": nom, "reference_reads": [],
            "rows": [{"match": "J03WQQ", "half": h, "condition": c, "policy_selection": refs["selection.json"]}
                     for h in replay.HALVES for c in replay.CONDITIONS]},
        "selection.json": {"kind": "wqq-campaign-calibration-v1", "run_id": replay.STUDY_NAME,
            "protocol": pp, "producer_held": "full", "selected": copy.deepcopy(replay.POLICY),
            "selected_id": replay.POLICY["id"], "policy_shared_across_conditions": True,
            "calibration_matches": ["J03WPY"], "prediction_seal": calref},
        "cal-seal.json": {"stage": "calibration", "protocol": pp, "run_id": replay.STUDY_NAME,
            "producer_held": "full", "reference_reads": [],
            "rows": [{"match": "J03WPY", "half": h, "condition": c}
                     for h in replay.HALVES for c in replay.CONDITIONS]},
        "parent-cache-admission-v1.json": {"scientific_change": False, "reference_reads": [],
            "repair_receipt": refs["parent-cache-repair-v1.json"], "admission": {"protocol": pp, "nomination": nom}},
        "parent-cache-repair-v1.json": {"study_protocol": pp, "reference_reads": [],
            "transform": {"scientific_change": False}},
    }
    monkeypatch.setattr(replay, "metadata", lambda ref: db[ref["path"]])
    monkeypatch.setattr(replay.chain, "checked", lambda ref: Path(ref["path"]))
    monkeypatch.setattr(replay.base, "verify_sources", lambda sources: None)
    return db


def test_completed_gate_checks_receipts_without_reading_metric_values(monkeypatch):
    db = candidate_tree(monkeypatch)
    gate, seals = replay.completed_candidate()
    assert gate["metric_values_used"] is False and gate["reference_arrays_opened"] == []
    assert set(seals) == {"J03WPY", "J03WQQ"}
    db["full-audit.json"]["score"] = "not a metric"
    assert replay.completed_candidate()[0] == gate


@pytest.mark.parametrize("record,field,value", [
    ("transition-scale-v1.json", "eligible", False),
    ("transition-scale-v1.json", "recipe_sha256", "changed"),
    ("transition-scale-v1.json", "WQQ_reference_reads", ["illegal"]),
    ("full-admission.json", "nomination", {"wrong": True}),
    ("full-audit.json", "prediction_seal", {"wrong": True}),
    ("full-audit.json", "WQQ_fitted_or_calibrated", True),
    ("selection.json", "selected", {"id": "different"}),
    ("selection.json", "calibration_matches", ["J03WPY", "J03WQQ"]),
    ("full-seal.json", "rows", []),
    ("cal-seal.json", "rows", []),
    ("parent-cache-repair-v1.json", "transform", {"scientific_change": True}),
])
def test_completed_gate_rejects_noncanonical_or_incomplete_lifecycle(monkeypatch, record, field, value):
    db = candidate_tree(monkeypatch); db[record][field] = value
    with pytest.raises(PermissionError):
        replay.completed_candidate()


def pilot(tmp_path, monkeypatch):
    relocate(tmp_path, monkeypatch)
    work = tmp_path / "work"; work.mkdir()
    gate = {"artifacts": {"transition-scale-v1.json": {"path": "nom.json", "sha256": "nom"},
                          "full-audit.json": {"path": "full-audit.json", "sha256": "full"}}}
    a = arrays(); expected = tmp_path / "expected.npz"; np.savez(expected, **a)
    fixtures = {}
    for match in ("J03WPY", "J03WQQ"):
        for h in replay.HALVES:
            p = tmp_path / f"private/improvement/external/prepared/{match}-{h}.npz"
            p.parent.mkdir(parents=True, exist_ok=True); np.savez(p, **raw())
            for c in replay.CONDITIONS:
                fixtures[f"{match}-{h}-{c}"] = {"match": match, "half": h, "condition": c,
                    "raw_input": replay.chain.pin(p), "expected": {name: {"artifact": replay.chain.pin(expected), "array": name} for name in a}}
    p = {"kind": replay.KIND, "policy": replay.POLICY, "foundation": replay.FOUNDATION,
         "gate": gate, "spec": {"runtime_pins": [], "compiled_kernels": {"loaders": {}}},
         "sources": {}, "runtime": {}, "fixtures": fixtures, "comparison_receipts": [],
         "reference_reads": [], "new_candidate": False, "policy_selection": False}
    replay.chain.immutable_json(work / "protocol.json", p)
    monkeypatch.setattr(replay, "foundation", lambda: {"spec": p["spec"]})
    monkeypatch.setattr(replay, "completed_candidate", lambda: (gate, {}))
    monkeypatch.setattr(replay, "sources", lambda: {})
    monkeypatch.setattr(replay.chain, "runtime_identity", lambda: {})
    monkeypatch.setattr(replay.chain, "compiled_loaders", lambda sources: {})
    monkeypatch.setattr(replay.chain, "verify_kernel_inventory", lambda spec: None)
    monkeypatch.setattr(replay, "comparison_inventory", lambda *args: (fixtures, []))
    calls = []
    class Bundle:
        def parent_from_raw(self, v):
            with pytest.raises(PermissionError):
                expected.read_bytes()
            calls.append(replay.chain.raw_identity(v))
            return a["parent"].copy(), {}, {}
        def native_from_raw(self, v):
            return a["logits"].copy()
    monkeypatch.setattr(replay.chain, "load_bundle", lambda spec: Bundle())
    return work, p, expected, calls


def test_full_synthetic_sequence_blocks_wqq_until_wpy_exact_and_resumes(tmp_path, monkeypatch):
    work, p, expected, calls = pilot(tmp_path, monkeypatch)
    load = np.load; opened = []
    def spy(path, *args, **kwargs):
        opened.append(Path(path)); return load(path, *args, **kwargs)
    monkeypatch.setattr(replay.np, "load", spy)
    with pytest.raises(FileNotFoundError):
        replay.predict(work, "J03WQQ")
    assert opened == [] and calls == []
    replay.predict(work, "J03WPY")
    assert len(calls) == 4 and expected not in opened
    result = replay.compare(work, "J03WPY")
    assert result["all_arrays_bit_exact"] is True and expected in opened
    opened.clear()
    replay.predict(work, "J03WQQ")
    assert len(calls) == 8 and expected not in opened
    snapshots = {f: (f.read_bytes(), f.stat().st_mtime_ns) for f in (work / "fresh").iterdir()}
    replay.predict(work, "J03WQQ")
    assert len(calls) == 8
    assert {f: (f.read_bytes(), f.stat().st_mtime_ns) for f in snapshots} == snapshots
    result = replay.compare(work, "J03WQQ")
    assert result["all_arrays_bit_exact"] is True and result["reference_reads"] == []
    assert result["metric_computation"] is False
    assert all(set(row) == {"parent", "logits", "transition"} for row in result["views"].values())


def test_comparison_needs_all_fresh_views_before_any_expected_open(tmp_path, monkeypatch):
    work, p, expected, _ = pilot(tmp_path, monkeypatch)
    replay.predict(work, "J03WPY", half="firstHalf")
    expected.unlink()
    with pytest.raises(FileNotFoundError, match="fresh-J03WPY-seal"):
        replay.compare(work)


@pytest.mark.parametrize("field,value", [("condition", "clean"), ("corruption_seed", -1), ("admission", {"wrong": 1})])
def test_misrouted_receipt_rejected_before_comparison_arrays(tmp_path, monkeypatch, field, value):
    work, p, expected, _ = pilot(tmp_path, monkeypatch)
    replay.predict(work, "J03WPY")
    key = "J03WPY-firstHalf-corrupted"; receipt = work / "fresh" / (key + ".json")
    m = replay.chain.read(receipt); m["identity"][field] = value; json_replace(receipt, m)
    seal_path = work / "fresh-J03WPY-seal.json"; seal = replay.chain.read(seal_path)
    seal["views"][key] = replay.chain.pin(receipt); json_replace(seal_path, seal)
    expected.unlink()
    with pytest.raises(ValueError, match="identity"):
        replay.compare(work)


@pytest.mark.parametrize("what", ["raw", "model", "policy", "expected_binding"])
def test_same_path_mutation_or_protocol_rebinding_rejected(tmp_path, monkeypatch, what):
    work, p, expected, _ = pilot(tmp_path, monkeypatch)
    if what == "raw":
        row = p["fixtures"][replay.keys("J03WPY")[0]]
        (tmp_path / row["raw_input"]["path"]).write_bytes(b"changed")
        with pytest.raises(ValueError, match="Changed pinned"):
            replay.predict(work, "J03WPY")
    elif what == "model":
        model = tmp_path / "model.pt"; model.write_bytes(b"frozen")
        ref = replay.chain.pin(model); model.write_bytes(b"mutated")
        with pytest.raises(ValueError, match="Changed pinned"):
            replay.chain.checked(ref)
    else:
        changed = copy.deepcopy(p)
        if what == "policy":
            changed["policy"]["alpha"] = .25
        else:
            changed["fixtures"][replay.keys("J03WPY")[0]]["expected"]["transition"]["array"] = "other"
        json_replace(work / "protocol.json", changed)
        with pytest.raises(ValueError):
            replay.protocol_for(work)


def test_commit_interruption_preserves_orphan_and_committed_bytes(tmp_path, monkeypatch):
    relocate(tmp_path, monkeypatch)
    work = tmp_path / "work"; ident = {"raw": "one"}; a = arrays(10)
    publish = replay.chain.immutable_json
    monkeypatch.setattr(replay.chain, "immutable_json", lambda *args: (_ for _ in ()).throw(RuntimeError("killed")))
    with pytest.raises(RuntimeError):
        replay.commit(work, "key", ident, a, {})
    orphan = next((work / "fresh").glob("*.npz")); before = orphan.read_bytes()
    monkeypatch.setattr(replay.chain, "immutable_json", publish)
    m = replay.commit(work, "key", ident, a, {})
    assert replay.chain.checked(m["output"]) != orphan and orphan.read_bytes() == before
    paths = [work / "fresh/key.json", replay.chain.checked(m["output"])]
    snapshots = [(f.read_bytes(), f.stat().st_mtime_ns) for f in paths]
    replay.commit(work, "key", ident, {}, {})
    assert [(f.read_bytes(), f.stat().st_mtime_ns) for f in paths] == snapshots
    with pytest.raises(ValueError):
        replay.commit(work, "key", {"raw": "two"}, a, {})
    paths[1].write_bytes(b"changed")
    with pytest.raises(ValueError):
        replay.committed(paths[0], ident)


def test_boundary_allows_only_registered_raw_wqq_and_blocks_truth_cache_audit(tmp_path, monkeypatch):
    relocate(tmp_path, monkeypatch)
    names = ["private/improvement/external/prepared/J03WQQ-firstHalf.npz", "cache/J03WQQ.npz",
             "cache/J03WPY.npz", "raw/J03WQQ-firstHalf.truth.npy", "raw/J03WR9.npz", "full-audit.json"]
    paths = [tmp_path / name for name in names]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"synthetic")
    with replay.ReadBoundary(paths, "J03WQQ"):
        assert paths[0].read_bytes() == b"synthetic"
        for p in paths[1:]:
            with pytest.raises(PermissionError):
                p.read_bytes()
        with pytest.raises(PermissionError):
            paths[0].write_bytes(b"write")
    with replay.ReadBoundary(paths, "J03WPY"):
        with pytest.raises(PermissionError):
            paths[0].read_bytes()
    with replay.ReadBoundary(paths, "J03WQQ", comparison=True):
        assert paths[1].read_bytes() == b"synthetic"
        for p in paths[3:]:
            with pytest.raises(PermissionError):
                p.read_bytes()


@pytest.mark.parametrize("bits", [(0x00000000, 0x80000000), (0x7FC00001, 0x7FC00002)])
def test_literal_byte_parity_distinguishes_signed_zero_and_nan_payloads(bits):
    a, b = [np.array([value], np.uint32).view(np.float32) for value in bits]
    result = replay.chain.compare_arrays(a, b)
    assert result["bit_exact"] is False and result["value_equal"] is True
    assert result["actual"] != result["expected"] and result["different_values"] == 0


def test_prepare_never_loads_arrays_and_pins_only_new_manifest(tmp_path, monkeypatch):
    work, protocol, _, _ = pilot(tmp_path, monkeypatch)
    new = tmp_path / "new"
    monkeypatch.setattr(replay.np, "load", lambda *args, **kwargs: pytest.fail("prepare loaded arrays"))
    result = replay.prepare(new)
    assert result["fixtures"] == protocol["fixtures"]
    assert result["new_candidate"] is False and result["policy_selection"] is False
    assert replay.prepare(new) == result


def test_fixture_seed_and_extra_field_guard(tmp_path, monkeypatch):
    work, protocol, _, _ = pilot(tmp_path, monkeypatch)
    row = protocol["fixtures"]["J03WPY-firstHalf-corrupted"]
    assert replay.chain.raw_identity(replay.fixture_raw(row)) == replay.chain.raw_identity(replay.chain.retained.corruption(raw(), 100))
    p = tmp_path / row["raw_input"]["path"]
    np.savez(p, **raw(), truth=np.zeros(200))
    with pytest.raises(ValueError, match="Changed pinned"):
        replay.fixture_raw(row)
    row = {**row, "raw_input": replay.chain.pin(p)}
    with pytest.raises(ValueError, match="exactly"):
        replay.fixture_raw(row)


def test_wqq_gate_rejects_incomplete_array_checks_and_changed_wpy_payload(tmp_path, monkeypatch):
    work, p, _, _ = pilot(tmp_path, monkeypatch)
    replay.predict(work); replay.compare(work)
    comparison_path = work / "comparison-J03WPY.json"
    original = replay.chain.read(comparison_path); changed = copy.deepcopy(original)
    del changed["views"][replay.keys("J03WPY")[0]]["logits"]
    json_replace(comparison_path, changed)
    with pytest.raises(PermissionError, match="omitted"):
        replay.wqq_admission(work, p)
    json_replace(comparison_path, original)
    receipt = replay.chain.read(work / "fresh" / (replay.keys("J03WPY")[0] + ".json"))
    (tmp_path / receipt["output"]["path"]).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="Changed pinned"):
        replay.wqq_admission(work, p)


@pytest.mark.parametrize("what", ["source", "runtime", "kernel", "candidate"])
def test_protocol_revalidates_entire_execution_closure(tmp_path, monkeypatch, what):
    work, p, _, _ = pilot(tmp_path, monkeypatch)
    if what == "source":
        monkeypatch.setattr(replay, "sources", lambda: {"source.py": "mutated"})
    elif what == "runtime":
        monkeypatch.setattr(replay.chain, "runtime_identity", lambda: {"threads": "different"})
    elif what == "kernel":
        def changed_kernel(spec):
            raise ValueError("kernel bytes changed")
        monkeypatch.setattr(replay.chain, "verify_kernel_inventory", changed_kernel)
    else:
        monkeypatch.setattr(replay, "completed_candidate", lambda: ({"wrong": "candidate"}, {}))
    with pytest.raises((ValueError, PermissionError)):
        replay.protocol_for(work)


def test_bundle_loading_and_prediction_share_no_cache_or_audit_read_grant(tmp_path, monkeypatch):
    work, p, expected, _ = pilot(tmp_path, monkeypatch)
    audit = tmp_path / "full-audit.json"; audit.write_text("synthetic")
    a = arrays()
    def check_closed():
        for path in (expected, audit):
            with pytest.raises(PermissionError):
                path.read_bytes()
    def loader(spec):
        check_closed()
        def parent(v):
            check_closed(); return a["parent"], {}, {}
        def native(v):
            check_closed(); return a["logits"]
        return NS(parent_from_raw=parent, native_from_raw=native)
    monkeypatch.setattr(replay.chain, "load_bundle", loader)
    replay.predict(work, "J03WPY")


def test_incomplete_wqq_seal_blocks_expected_array_open(tmp_path, monkeypatch):
    work, p, expected, _ = pilot(tmp_path, monkeypatch)
    replay.predict(work); replay.compare(work)
    replay.predict(work, "J03WQQ", half="firstHalf")
    loaded = []; original = np.load
    def spy(path, *args, **kwargs):
        loaded.append(Path(path)); return original(path, *args, **kwargs)
    monkeypatch.setattr(replay.np, "load", spy)
    with pytest.raises(FileNotFoundError, match="fresh-J03WQQ-seal"):
        replay.compare(work, "J03WQQ")
    assert expected not in loaded


@pytest.mark.parametrize("mutation", ["result_flag", "array_sha", "all_flag"])
def test_wqq_rejects_nonexact_wpy_comparison_before_raw_load(tmp_path, monkeypatch, mutation):
    work, p, _, _ = pilot(tmp_path, monkeypatch)
    replay.predict(work); replay.compare(work)
    path = work / "comparison-J03WPY.json"; r = replay.chain.read(path)
    row = r["views"][replay.keys("J03WPY")[0]]["transition"]
    if mutation == "result_flag":
        row["bit_exact"] = False
    elif mutation == "array_sha":
        row["actual"]["sha256"] = "different"
    else:
        r["all_arrays_bit_exact"] = False
    json_replace(path, r)
    monkeypatch.setattr(replay.np, "load", lambda *args, **kwargs: pytest.fail("WQQ raw opened before admission"))
    with pytest.raises(PermissionError):
        replay.predict(work, "J03WQQ")


def test_expected_inventory_binds_raw_models_policy_and_seals_without_payload_reads(tmp_path, monkeypatch):
    relocate(tmp_path, monkeypatch)
    study = tmp_path / "study"; monkeypatch.setattr(replay, "STUDY", study)
    folder = study / "views/full"; folder.mkdir(parents=True)
    pp = {"path": "protocol.json", "sha256": "protocol"}
    nom = {"path": "nomination.json", "sha256": "nomination"}
    gate = {"artifacts": {"protocol.json": pp, "transition-scale-v1.json": nom}}
    spec = {"models": {"native": {"path": "native.pt", "sha256": "native"},
                       "bridge": {"path": "context.pt", "sha256": "context"}}}
    sealed = {m: {"rows": []} for m in ("J03WPY", "J03WQQ")}
    native_paths = []
    payload = {"path": "unopened.npz", "sha256": "not-loaded"}
    for match in sealed:
        for h in replay.HALVES:
            for c in replay.CONDITIONS:
                key = f"{match}-{h}-{c}"
                raw_pin = {"path": f"private/improvement/external/prepared/{match}-{h}.npz", "sha256": "raw"}
                ident = {"which": "full", "match": match, "half": h, "condition": c}
                native = {**ident, "reference_reads": [], "family": "v70", "permit": nom,
                    "corruption_seed": 100 if c == "corrupted" else None, "output": payload, "raw_input": raw_pin,
                    "model_bundle": {"model": spec["models"]["native"], "context_model": spec["models"]["bridge"],
                                     "fit_matches": list(replay.base.TRAIN)}}
                parent = {**ident, "reference_arrays_opened": [], "artifact": payload, "nomination": nom,
                          "fit_matches": list(replay.base.TRAIN), "calibration_matches": [*replay.base.TRAIN, "J03WPY"]}
                npth, ppth = folder / (key + "-native.json"), folder / (key + "-parent.json")
                json_replace(npth, native); json_replace(ppth, parent); native_paths.append(npth)
                selected = replay.POLICY if match == "J03WPY" else {**replay.POLICY, "id": "candidate"}
                view = {"identity": {**ident, "protocol": pp, "policies": [selected], "input_identity": {
                    "logit_receipt": replay.chain.pin(npth), "parent_receipt": replay.chain.pin(ppth),
                    "logits": payload, "parent": payload, "raw_input": raw_pin}},
                    "reference_reads": [], "artifact": payload, "arrays": {selected["id"]: selected["id"]}}
                json_replace(folder / (key + ".json"), view)
                sealed[match]["rows"].append({**ident, "artifact": payload, "arrays": view["arrays"]})
    monkeypatch.setattr(replay.np, "load", lambda *args, **kwargs: pytest.fail("inventory parsed predictions"))
    fixtures, receipts = replay.comparison_inventory(spec, gate, sealed)
    assert len(fixtures) == 8 and len(receipts) == 24
    path = native_paths[0]; native = replay.chain.read(path)
    native["model_bundle"]["context_model"] = {"path": "other.pt", "sha256": "other"}
    json_replace(path, native)
    view_path = folder / "J03WPY-firstHalf-clean.json"; view = replay.chain.read(view_path)
    view["identity"]["input_identity"]["logit_receipt"] = replay.chain.pin(path); json_replace(view_path, view)
    with pytest.raises(ValueError, match="identity, model"):
        replay.comparison_inventory(spec, gate, sealed)
