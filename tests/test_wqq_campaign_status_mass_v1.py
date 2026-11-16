"""Synthetic only: numerical ablation parity and cache/admission integration."""
import copy
import json

import numpy as np
import pytest
import torch

from scripts import wqq_campaign_status_mass_v1 as status
from test_wqq_campaign_gate import caches as cached_inputs


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def policy(alpha, mode="joint"):
    return next(p for p in status.policy_grid() if p["alpha"] == alpha and p["status_mode"] == mode)


def test_nine_fixed_policies_and_strict_pure_api():
    grid = status.policy_grid()
    assert len(grid) == len({p["id"] for p in grid}) == 9
    assert [p["alpha"] for p in grid[:5]] == list(status.base.ALPHAS)
    assert [p["status_mode"] for p in grid] == ["joint"] * 5 + ["parent_D"] * 4
    assert sum(p["unchanged"] for p in grid) == 1
    for p in grid:
        status.validate_policy({**p, "id": "candidate"})
    for changed in ({"condition": "clean"}, {"parent_confidence": .8}, {"alpha": .1}, {"status_mode": "learned"}):
        with pytest.raises(ValueError):
            status.validate_policy({**grid[0], **changed})


@pytest.mark.parametrize("alpha", status.base.ALPHAS)
def test_five_controls_are_bit_exact_including_parent_x_and_zero(alpha):
    rng = np.random.default_rng(21)
    parent = rng.integers(0, 4, 131, dtype=np.int8)
    logits = rng.normal(size=(131, 5)).astype(np.float32)
    actual = status.compose(parent, logits, policy(alpha))
    np.testing.assert_array_equal(actual, status.base.compose(parent, logits, alpha, .98))
    if alpha == 0:
        assert actual is not parent
        np.testing.assert_array_equal(actual, parent)


@pytest.mark.parametrize("alpha", status.base.ALPHAS[1:])
def test_parent_marginal_normalization_and_original_conditional_odds(alpha):
    rng = np.random.default_rng(55)
    parent = rng.integers(0, 4, 87, dtype=np.int8)
    logits = rng.normal(0, 30, (87, 5)).astype(np.float32)
    logits[0, :2] = [1e4, -1e4]; parent[0] = 2
    original, prior = status.mixed_nodes(parent, logits, alpha)
    nodes = status.parent_status_nodes(parent, logits, alpha)
    probabilities = np.exp(nodes)
    assert np.isfinite(nodes).all() and np.all(probabilities > 0)
    np.testing.assert_allclose(probabilities.sum(1), 1., atol=2e-15, rtol=0)
    np.testing.assert_allclose(probabilities[:, 2], prior[:, 2], atol=2e-15, rtol=0)
    expected_log_odds = original[:, 1].astype(np.float64) - original[:, 0].astype(np.float64)
    np.testing.assert_allclose(nodes[:, 1] - nodes[:, 0], expected_log_odds, atol=7e-15, rtol=0)
    if alpha == 1:
        assert probabilities[0, 0] < 1e-8  # A second clip would destroy the preserved odds.


def test_typed_hazards_unchanged_and_soft_status_is_not_a_label_lock(monkeypatch):
    parent = np.array([0, 2, 0], np.int8)
    logits = np.zeros((3, 5), np.float32); logits[:, 2:] = -20
    before = logits.copy()
    result = status.compose(parent, logits, policy(.25, "parent_D"))
    assert result[1] != parent[1]  # Coupled decoder may remove this isolated dead frame.
    captured = []
    monkeypatch.setattr(status.base.native, "decode", lambda nodes, hazards: captured.append(hazards.copy()) or parent.copy())
    status.compose(parent, logits, policy(.5, "parent_D"))
    np.testing.assert_array_equal(captured[0], logits[:, 2:])
    np.testing.assert_array_equal(logits, before)


def test_team_symmetry_and_invalid_inputs():
    rng = np.random.default_rng(115)
    parent = rng.integers(0, 4, 97, dtype=np.int8)
    logits = rng.normal(size=(97, 5)).astype(np.float32)
    mapping = np.array([1, 0, 2, 3], np.int8)
    swapped = logits.copy(); swapped[:, 0] *= -1
    for p in status.policy_grid()[5:]:
        np.testing.assert_array_equal(status.compose(mapping[parent], swapped, p), mapping[status.compose(parent, logits, p)])
    bad = logits.copy(); bad[0, 0] = np.nan
    for p, z in ((parent.astype(float), logits), (parent[:-1], logits), (parent, bad)):
        with pytest.raises(ValueError):
            status.compose(p, z, policy(.25, "parent_D"))


@pytest.fixture
def caches(cached_inputs, monkeypatch):
    c = cached_inputs
    monkeypatch.setattr(status, "ROOT", c.root)
    monkeypatch.setattr(status, "source_pins", lambda: {})
    input_seal = c.root / "diagnostic-inputs.json"; input_seal.write_text('{}')
    report = c.root / "diagnostic.json"; report.write_text('{}')
    diagnosis = {"report": status.base.pin(report), "input_seal": status.base.pin(input_seal)}
    c.protocol["kind"] = status.KIND
    c.protocol["recipe"].pop("gate"); c.protocol["recipe"].pop("composition")
    c.protocol["recipe"].update(mechanism=status.mechanism(), policies=status.policy_grid(), diagnostic_motivation=diagnosis)
    c.diagnosis = diagnosis
    c.produce = lambda condition="clean", match="J03WPY", policies=None: status.produce_view(
        c.folder, "J03WMX", match, "firstHalf", condition, policies or status.policy_grid())
    return c


def test_cached_inputs_and_raw_identity_are_enforced_on_exact_resume(caches):
    entry = caches.create()
    first = caches.produce()
    assert caches.produce() == first and caches.raw_calls == []
    assert len(first["arrays"]) == 9 and first["reference_reads"] == []
    with np.load(status.base.checked(first["artifact"]), allow_pickle=False) as z:
        with np.load(status.base.checked(entry["parent"]), allow_pickle=False) as p:
            np.testing.assert_array_equal(z["joint_a000"], p["parent"])
    with status.base.checked(entry["raw_input"]).open("ab") as handle:
        handle.write(b"changed observations")
    with pytest.raises(ValueError, match="raw fixture"):
        caches.produce()


def test_full_wpy_ancestry_reaches_oof_and_changed_input_fails_before_truth(caches, monkeypatch):
    evaluation = status.study.evaluation
    for match in ("J03WPY", "J03WMX"):
        for half in status.base.HALVES:
            for condition in ("clean", "corrupted"):
                caches.create(match=match, half=half, condition=condition)
    original_create, original_load, original_select = evaluation.create_seal, evaluation.load_seal, evaluation.select_policy_from_seal
    monkeypatch.setattr(evaluation, "create_seal", lambda *a, **kw: original_create(*a, **{**kw, "root": caches.root}))
    monkeypatch.setattr(evaluation, "load_seal", lambda *a, **kw: original_load(*a, **{**kw, "root": caches.root}))
    calls = []
    def truth(m, h):
        calls.append((m, h)); return np.resize(np.array([0, 0, 1, 2, 3], np.int8), 33)
    monkeypatch.setattr(evaluation, "select_policy_from_seal", lambda *a, **kw: original_select(*a, **{**kw, "root": caches.root, "truth_loader": truth}))
    with status.installed():
        inventory = status.study.produce_scored_match(caches.folder, "J03WMX", "J03WMX")
    assert calls == [("J03WPY", h) for h in status.base.HALVES]
    assert len({json.dumps(row["policy_selection"], sort_keys=True) for row in inventory["rows"]}) == 1
    nodes = inventory["ancestry_nodes"]
    assert any(node["kind"] == "calibration" and node["calibration_matches"] == ["J03WPY"] for node in nodes.values())
    row = inventory["rows"][0]
    evaluation.validate_ancestry(nodes, row["ancestry_roots"], "J03WMX", root=caches.root)
    path = status.base.DATA / "J03WPY-firstHalf.npz"
    with path.open("ab") as handle:
        handle.write(b"changed after selection")
    with pytest.raises(ValueError, match="Changed immutable"):
        evaluation.validate_ancestry(nodes, row["ancestry_roots"], "J03WMX", root=caches.root)
    assert calls == [("J03WPY", h) for h in status.base.HALVES]


def test_registration_binds_diagnostic_and_all32_caches(caches, monkeypatch):
    for held in status.base.TRAIN:
        for match in (held, "J03WPY"):
            for h in status.base.HALVES:
                for c in ("clean", "corrupted"):
                    caches.create(held, match, h, c)
    monkeypatch.setattr(status, "diagnostic_evidence", lambda path: caches.diagnosis)
    monkeypatch.setattr(status.study.evaluation, "load_seal", lambda *a, **kw: (
        {"run_id": caches.source.name, "protocol": status.base.pin(caches.source / "protocol.json")}, None))
    folder = caches.folder.with_name("registered-status")
    protocol = status.register(folder, caches.source, caches.root / "diagnostic.json")
    assert len(protocol["cache_index"]) == 32 and len(protocol["recipe"]["policies"]) == 9
    assert protocol["recipe"]["diagnostic_motivation"] == caches.diagnosis
    assert status.register(folder, caches.source, caches.root / "diagnostic.json") == protocol


def test_diagnostic_receipt_must_be_train_only_and_its_inputs_unchanged(caches, monkeypatch):
    source = caches.root / "diagnostic-source.py"; source.write_text("# synthetic source")
    input_path = caches.root / "diagnostic-inputs.json"
    status.base.atomic_json(input_path, {"kind": "wqq-v70-error-diagnostic-inputs-v1", "reference_reads": [],
                                       "inputs": [status.base.pin(source)]})
    report = {"kind": "wqq-v70-error-diagnostic-v1", "diagnostic_only": True, "fits_or_decodes": 0,
              "input_seal": status.base.pin(input_path),
              "reference_reads": [{"match": m, "half": h} for m in status.base.TRAIN for h in status.base.HALVES]}
    report_path = caches.root / "diagnostic.json"; status.base.atomic_json(report_path, report)
    monkeypatch.setattr(status.study.evaluation, "checked", lambda ref: status.base.checked(ref))
    assert status.diagnostic_evidence(report_path)["report"] == status.base.pin(report_path)
    bad = copy.deepcopy(report); bad["reference_reads"][0]["match"] = "J03WPY"
    status.base.atomic_json(report_path, bad)
    with pytest.raises(ValueError, match="TRAIN4-only"):
        status.diagnostic_evidence(report_path)
    status.base.atomic_json(report_path, report); source.write_text("# changed")
    with pytest.raises(ValueError, match="hash-bound"):
        status.diagnostic_evidence(report_path)


def test_wrong_full_nomination_fails_before_any_inference(caches):
    caches.create()
    for match in ("J03WPY", "J03WQQ"):
        with pytest.raises(PermissionError):
            status.produce_view(caches.folder, "full", match, "firstHalf", "clean", status.policy_grid())
    permit = caches.root / "permit.json"
    value = {"candidate_id": caches.folder.name, "protocol": status.base.pin(caches.folder / "protocol.json"),
             "selected_recipe": copy.deepcopy(caches.protocol["recipe"])}
    value["selected_recipe"]["policies"] = status.policy_grid()[:5]
    permit.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="exact study"):
        status.full_audit(caches.folder, permit)
    assert not (caches.folder / "full-admission.json").exists() and caches.raw_calls == []


def test_scoped_adapter_restores_after_failure():
    original = status.study.produce_view
    with pytest.raises(RuntimeError):
        with status.installed():
            assert status.study.produce_view is status.produce_view
            raise RuntimeError("synthetic")
    assert status.study.produce_view is original


def test_source_manifest_pins_versioned_adapter_test_and_diagnostic():
    sources = status.source_pins()
    for name in ("scripts/wqq_campaign_status_mass_v1.py", "tests/test_wqq_campaign_status_mass_v1.py",
                 "scripts/wqq_campaign_error_diagnostic_v1.py", "scripts/wqq_campaign_gate.py"):
        assert sources[name] == status.base.sha256(status.ROOT / name)
