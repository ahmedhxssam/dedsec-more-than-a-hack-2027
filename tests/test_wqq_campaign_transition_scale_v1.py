"""Synthetic transition-energy, provenance, shared-policy and admission tests."""
import copy
import itertools
import json

import numpy as np
import pytest
import torch

from scripts import wqq_campaign_transition_scale_v1 as scale
from test_wqq_campaign_status_mass_v1 import caches as status_caches, cached_inputs


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def policy(value=1., alpha=.5, mode="joint"):
    return next(p for p in scale.policy_grid() if p["transition_scale"] == value
                and p["alpha"] == alpha and p["status_mode"] == mode)


def test_exact27_registry_pure_api_and_original_tie_preference(monkeypatch):
    grid = scale.policy_grid()
    assert len(grid) == len({p["id"] for p in grid}) == 27
    assert [p["transition_scale"] for p in grid] == [1.] * 9 + [.5] * 9 + [2.] * 9
    assert sum(p["unchanged"] for p in grid) == 3
    for p in grid:
        scale.validate_policy({**p, "id": "candidate"})
    for change in ({"condition": "clean"}, {"transition_scale": .75}, {"parent_confidence": .8},
                   {"alpha": .1}, {"status_mode": "learned"}):
        with pytest.raises(ValueError):
            scale.validate_policy({**grid[0], **change})
    evaluation = scale.study.evaluation
    monkeypatch.setattr(evaluation, "official_metrics", lambda: None)
    monkeypatch.setattr(evaluation, "score_full", lambda *args: ({"spell_f1_mean": .7, "macro_f1": .9}, {}))
    # Even a baseline parent_D policy beats a scaled joint policy on exact ties.
    policies = [policy(.5), policy(1., mode="parent_D")]
    cases = {(h, c): {p["id"]: np.zeros(4, np.int8) for p in policies}
             for h in evaluation.HALVES for c in evaluation.CONDITIONS}
    selected = evaluation.select_policy(cases, {h: np.zeros(4, np.int8) for h in evaluation.HALVES}, policies)
    assert selected["selected"]["transition_scale"] == 1.


@pytest.mark.parametrize("index", range(9))
def test_scale1_calls_exact_original_status_composition(index, monkeypatch):
    rng = np.random.default_rng(330)
    parent = rng.integers(0, 4, 93, dtype=np.int8)
    logits = rng.normal(size=(93, 5)).astype(np.float32)
    original = scale.status.policy_grid()[index]
    expected = scale.status.compose(parent, logits, original)
    calls, implementation = [], scale.status.compose
    monkeypatch.setattr(scale.status, "compose", lambda p, z, q: calls.append(q) or implementation(p, z, q))
    monkeypatch.setattr(scale, "decode", lambda *a: pytest.fail("Scale1 bypassed original compose"))
    np.testing.assert_array_equal(scale.compose(parent, logits, scale.policy_grid()[index]), expected)
    assert calls == [original]


def test_all_alpha_zero_policies_are_exact_parent_and_never_decode(monkeypatch):
    parent = np.array([0, 1, 2, 3, 0], np.int8)
    logits = np.ones((5, 5), np.float32)
    monkeypatch.setattr(scale, "decode", lambda *a: pytest.fail("Alpha0 decoded"))
    monkeypatch.setattr(scale.base.native, "decode", lambda *a: pytest.fail("Alpha0 decoded"))
    for value in scale.SCALES:
        result = scale.compose(parent, logits, policy(value, 0.))
        assert result is not parent
        np.testing.assert_array_equal(result, parent)


@pytest.mark.parametrize("value", scale.SCALES)
def test_transition_energy_includes_restart_log2_and_bruteforce_path_parity(value):
    rng = np.random.default_rng(61)
    hazard = rng.normal(size=(5, 3))
    nodes = rng.normal(size=(5, 3))
    transition = scale.transition_scores(hazard, value)
    cost = np.logaddexp(0., -hazard)
    np.testing.assert_array_equal(transition[:, 0, 1], -value * cost[:, 0])
    np.testing.assert_array_equal(transition[:, 0, 2], -value * cost[:, 1])
    np.testing.assert_array_equal(transition[:, 2, 0], value * (-cost[:, 2] - np.log(2.)))
    np.testing.assert_array_equal(transition[:, 2, 1], transition[:, 2, 0])
    assert np.all(transition <= 0) and np.count_nonzero(transition[:, np.arange(3), np.arange(3)]) == 0
    def score(path):
        return sum(nodes[i, label] / 5. for i, label in enumerate(path)) + sum(
            transition[i, path[i-1], path[i]] for i in range(1, len(path)))
    expected = max(itertools.product(range(3), repeat=5), key=score)
    np.testing.assert_array_equal(scale.decode(nodes, hazard, value), expected)
    if value == 1:
        np.testing.assert_array_equal(scale.decode(nodes, hazard, value), scale.base.native.decode(nodes, hazard))


def test_lower_cost_can_restore_supported_short_spell_without_changing_nodes():
    nodes = np.array([[0., -100., -100.], [-8., 0., -100.], [0., -100., -100.]])
    hazard = np.full((3, 3), -np.log(np.expm1(1.)))
    np.testing.assert_array_equal(scale.decode(nodes, hazard, .5), [0, 1, 0])
    for value in (1., 2.):
        np.testing.assert_array_equal(scale.decode(nodes, hazard, value), [0, 0, 0])


def test_team_symmetry_no_input_mutation_and_invalid_inputs():
    rng = np.random.default_rng(777)
    parent = rng.integers(0, 4, 127, dtype=np.int8)
    logits = rng.normal(size=(127, 5)).astype(np.float32); before = logits.copy()
    mapping = np.array([1, 0, 2, 3], np.int8)
    swapped = logits.copy(); swapped[:, 0] *= -1
    for p in scale.policy_grid():
        np.testing.assert_array_equal(scale.compose(mapping[parent], swapped, p), mapping[scale.compose(parent, logits, p)])
    np.testing.assert_array_equal(logits, before)
    for value in (.5, 1., 2.):
        bad = logits.copy(); bad[0, 0] = np.nan
        with pytest.raises(ValueError): scale.compose(parent, bad, policy(value))
    for nodes, hazards in ((np.ones((0, 3)), np.ones((0, 3))), (np.ones((3, 3)), np.ones((2, 3))),
                           (np.ones((3, 3)), np.full((3, 3), np.inf))):
        with pytest.raises(ValueError): scale.decode(nodes, hazards, .5)


@pytest.fixture
def caches(status_caches, monkeypatch):
    c = status_caches
    monkeypatch.setattr(scale, "ROOT", c.root)
    monkeypatch.setattr(scale, "source_pins", lambda: {})
    c.control_folder = c.folder.with_name("status-control"); c.control_folder.mkdir()
    control = {}
    for key, name in (("protocol", "protocol.json"), ("oof_audit", "oof-audit.json"), ("oof_seal", "oof-seal.json")):
        path = c.control_folder / name; path.write_text('{}'); control[key] = scale.base.pin(path)
    c.control = control
    c.protocol["kind"] = scale.KIND
    c.protocol["recipe"].update(mechanism=scale.mechanism(), policies=scale.policy_grid(), status_control=control)
    c.produce = lambda condition="clean", match="J03WPY", policies=None: scale.produce_view(
        c.folder, "J03WMX", match, "firstHalf", condition, policies or scale.policy_grid())
    return c


def test_cached_inputs_and_raw_identity_are_enforced_on_exact_resume(caches):
    entry = caches.create()
    first = caches.produce()
    assert caches.produce() == first and caches.raw_calls == []
    assert len(first["arrays"]) == 27 and first["reference_reads"] == []
    with np.load(scale.base.checked(first["artifact"]), allow_pickle=False) as z:
        with np.load(scale.base.checked(entry["parent"]), allow_pickle=False) as p:
            for value in scale.SCALES:
                np.testing.assert_array_equal(z[policy(value, 0.)["id"]], p["parent"])
    with scale.base.checked(entry["raw_input"]).open("ab") as handle: handle.write(b"changed")
    with pytest.raises(ValueError, match="raw fixture"): caches.produce()


def test_complete_wpy_ancestry_shared_policy_and_seal_before_truth(caches, monkeypatch):
    evaluation = scale.study.evaluation
    for match in ("J03WPY", "J03WMX"):
        for half in scale.base.HALVES:
            for condition in ("clean", "corrupted"): caches.create(match=match, half=half, condition=condition)
    original_create, original_load, original_select = evaluation.create_seal, evaluation.load_seal, evaluation.select_policy_from_seal
    monkeypatch.setattr(evaluation, "create_seal", lambda *a, **kw: original_create(*a, **{**kw, "root": caches.root}))
    monkeypatch.setattr(evaluation, "load_seal", lambda *a, **kw: original_load(*a, **{**kw, "root": caches.root}))
    calls = []
    def truth(m, h):
        assert (caches.folder / "calibration/J03WMX/seal.json").exists()
        calls.append((m, h)); return np.resize(np.array([0, 0, 1, 2, 3], np.int8), 33)
    monkeypatch.setattr(evaluation, "select_policy_from_seal", lambda *a, **kw: original_select(*a, **{**kw, "root": caches.root, "truth_loader": truth}))
    with scale.installed(): inventory = scale.study.produce_scored_match(caches.folder, "J03WMX", "J03WMX")
    assert calls == [("J03WPY", h) for h in scale.base.HALVES]
    assert len({json.dumps(row["policy_selection"], sort_keys=True) for row in inventory["rows"]}) == 1
    nodes = inventory["ancestry_nodes"]; row = inventory["rows"][0]
    assert any(n["kind"] == "calibration" and n["calibration_matches"] == ["J03WPY"] for n in nodes.values())
    evaluation.validate_ancestry(nodes, row["ancestry_roots"], "J03WMX", root=caches.root)
    with (scale.base.DATA / "J03WPY-firstHalf.npz").open("ab") as handle: handle.write(b"changed")
    with pytest.raises(ValueError, match="Changed immutable"):
        evaluation.validate_ancestry(nodes, row["ancestry_roots"], "J03WMX", root=caches.root)
    assert len(calls) == 2


def test_registration_binds_train_diagnostic_status_control_and32_inputs(caches, monkeypatch):
    for held in scale.base.TRAIN:
        for match in (held, "J03WPY"):
            for half in scale.base.HALVES:
                for condition in ("clean", "corrupted"): caches.create(held, match, half, condition)
    monkeypatch.setattr(scale, "diagnostic_evidence", lambda path: caches.diagnosis)
    monkeypatch.setattr(scale, "status_evidence", lambda path: (caches.control, caches.protocol))
    monkeypatch.setattr(scale.study.evaluation, "load_seal", lambda *a, **kw: (
        {"run_id": caches.source.name, "protocol": scale.base.pin(caches.source / "protocol.json")}, None))
    folder = caches.folder.with_name("registered-scale")
    result = scale.register(folder, caches.source, caches.root / "diagnostic.json", caches.control_folder)
    assert len(result["cache_index"]) == 32 and len(result["recipe"]["policies"]) == 27
    assert result["recipe"]["status_control"] == caches.control
    assert result["recipe"]["diagnostic_motivation"] == caches.diagnosis
    assert scale.register(folder, caches.source, caches.root / "diagnostic.json", caches.control_folder) == result
    with scale.base.checked(caches.control["oof_audit"]).open("ab") as handle: handle.write(b"changed")
    with pytest.raises(ValueError, match="hash-bound"): scale.protocol_for(folder)


def synthetic_audit(candidate=.76):
    row = {"candidate": candidate, "v70": .731, "v69": .735}
    return {"stage": "oof", "reference_reads": [{"match": m, "half": h, "sha256": "synthetic"}
            for m in scale.base.TRAIN for h in scale.base.HALVES],
            "overall": dict(row), "macro_f1": dict(row), "control_seals": {},
            "conditions": {c: dict(row) for c in ("clean", "corrupted")},
            "per_match": {m: dict(row) for m in scale.base.TRAIN},
            "per_producer_pair": {str(i): dict(row) for i in (0, 1)},
            "details": {m: {c: {"candidate": {"spell_f1_mean": candidate}} for c in ("clean", "corrupted")} for m in scale.base.TRAIN}}


def test_parent_control_gate_is_strict_overall_only_and_all_regressions_visible():
    parent = synthetic_audit(.742); candidate = synthetic_audit(.75)
    candidate["conditions"]["clean"]["candidate"] = .73
    candidate["per_producer_pair"]["1"]["candidate"] = .73
    evidence = {"protocol": {}, "oof_seal": {}, "oof_audit": {}}
    result = scale.attach_parent_control(candidate, parent, evidence)
    assert result["eligible"] is True
    assert result["delta_vs_status_mass"]["conditions"]["clean"] < 0
    assert result["delta_vs_status_mass"]["per_producer_pair"]["1"] < 0
    assert result["overall"]["status_mass"] == .742 and "status_mass" not in candidate["overall"]
    assert scale.attach_parent_control(result, parent, evidence) == result
    for value in (.74, .742):
        assert scale.attach_parent_control(synthetic_audit(value), parent, evidence)["eligible"] is False
    changed = copy.deepcopy(parent); changed["reference_reads"][0]["sha256"] = "other"
    with pytest.raises(ValueError, match="reference identities"): scale.attach_parent_control(candidate, changed, evidence)


def test_status_control_evidence_requires_exact_oof_stage_identity_and_seal(caches, monkeypatch):
    parent = synthetic_audit(.742)
    parent.update(kind="wqq-campaign-audit-v1", run_id=caches.control_folder.name,
                  protocol=caches.control["protocol"], prediction_seal=caches.control["oof_seal"])
    path = caches.control_folder / "oof-audit.json"
    scale.base.atomic_json(path, parent)
    monkeypatch.setattr(scale.status, "protocol_for", lambda folder: {"synthetic": True})
    calls = []
    def load(path, stage):
        calls.append(stage)
        return {"run_id": caches.control_folder.name, "protocol": caches.control["protocol"]}, None
    monkeypatch.setattr(scale.study.evaluation, "load_seal", load)
    evidence, protocol = scale.status_evidence(caches.control_folder)
    assert evidence["oof_audit"] == scale.base.pin(path) and protocol == {"synthetic": True}
    assert calls == ["oof"]
    for changed in ({"stage": "full"}, {"run_id": "other"}, {"per_match": {}}, {"protocol": {}}):
        scale.base.atomic_json(path, {**parent, **changed})
        with pytest.raises(ValueError, match="TRAIN4 OOF"): scale.status_evidence(caches.control_folder)
    assert calls == ["oof"]


def test_audit_writes_final_parent_gate_once_and_resume_never_scores_again(caches, monkeypatch):
    caches.create()
    parent = synthetic_audit(.742)
    parent_path = caches.control_folder / "oof-audit.json"
    scale.base.atomic_json(parent_path, parent)
    caches.control["oof_audit"] = scale.base.pin(parent_path)
    scale.base.atomic_json(caches.folder / "protocol.json", caches.protocol)
    monkeypatch.setattr(scale, "status_evidence", lambda path: (caches.control, {}))
    seal = caches.folder / "oof-seal.json"; seal.write_text('{}')
    monkeypatch.setattr(scale.study.evaluation, "load_seal", lambda *a, **kw: ({}, {}))
    calls = []
    def score(path):
        assert path == seal and path.exists()
        assert scale.study._TRUTH == set(scale.base.TRAIN)
        calls.append(path)
        result = synthetic_audit(.75)
        result.update(prediction_seal=scale.base.pin(seal), protocol=scale.base.pin(caches.folder / "protocol.json"))
        return result
    monkeypatch.setattr(scale.study.evaluation, "audit_predictions", score)
    result = scale.audit_oof(caches.folder)
    assert result["eligible"] and result["overall"]["status_mass"] == .742
    assert result == scale.base.read_json(caches.folder / "oof-audit.json")
    assert result == scale.audit_oof(caches.folder) and len(calls) == 1
    altered = {**result, "eligible": False}
    scale.base.atomic_json(caches.folder / "oof-audit.json", altered)
    with pytest.raises(ValueError, match="audit evidence or gate"): scale.audit_oof(caches.folder)
    assert len(calls) == 1


def test_wrong_full_nomination_and_failed_parent_gate_stop_before_inference(caches, monkeypatch):
    caches.create()
    for match in ("J03WPY", "J03WQQ"):
        with pytest.raises(PermissionError):
            scale.produce_view(caches.folder, "full", match, "firstHalf", "clean", scale.policy_grid())
    permit = caches.root / "permit.json"
    value = {"candidate_id": caches.folder.name, "protocol": scale.base.pin(caches.folder / "protocol.json"),
             "selected_recipe": copy.deepcopy(caches.protocol["recipe"])}
    value["selected_recipe"]["policies"] = scale.policy_grid()[:9]; permit.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="exact study"): scale.full_audit(caches.folder, permit)
    monkeypatch.setattr(scale.runtime, "preflight_full", lambda *args: {})
    audit_path = caches.folder / "oof-audit.json"; audit_path.write_text('{}')
    value["oof_audit"] = scale.base.pin(audit_path); permit.write_text(json.dumps(value))
    monkeypatch.setattr(scale, "audit_oof", lambda *args: {"eligible": False})
    with pytest.raises(PermissionError, match="passing"): scale.full_audit(caches.folder, permit)
    assert not (caches.folder / "full-admission.json").exists() and caches.raw_calls == []


def test_full_requires_canonical_completed_audit_before_gate_verification(caches, monkeypatch):
    caches.create()
    permit = caches.root / "permit.json"; permit.write_text('{"oof_audit": {}}')
    monkeypatch.setattr(scale.runtime, "preflight_full", lambda *args: {})
    monkeypatch.setattr(scale, "audit_oof", lambda *a: pytest.fail("Missing canonical audit triggered scoring"))
    with pytest.raises(PermissionError, match="passing"): scale.preflight_full(caches.folder, permit)
    audit = caches.folder / "oof-audit.json"; audit.write_text('{}')
    with pytest.raises(PermissionError, match="passing"): scale.preflight_full(caches.folder, permit)
    permit.write_text(json.dumps({"oof_audit": scale.base.pin(audit)}))
    monkeypatch.setattr(scale, "audit_oof", lambda *a: {"eligible": True})
    assert scale.preflight_full(caches.folder, permit) == {}


def test_scoped_adapter_and_source_manifest():
    original = scale.study.produce_view
    with pytest.raises(RuntimeError):
        with scale.installed():
            assert scale.study.produce_view is scale.produce_view
            raise RuntimeError("synthetic")
    assert scale.study.produce_view is original
    sources = scale.source_pins()
    for path in ("scripts/wqq_campaign_transition_scale_v1.py", "tests/test_wqq_campaign_transition_scale_v1.py",
                 "scripts/wqq_campaign_status_mass_v1.py", "scripts/wqq_campaign_error_diagnostic_v1.py"):
        assert sources[path] == scale.base.sha256(scale.ROOT / path)
