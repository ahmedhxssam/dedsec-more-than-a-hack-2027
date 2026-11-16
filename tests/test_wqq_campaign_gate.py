"""Synthetic cache/admission checks; no campaign fixtures or references opened."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import wqq_campaign_gate as gate


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def policy(a, b):
    return next(p for p in gate.policy_grid() if (p["alpha_observed"], p["alpha_missing"]) == (a, b))


def test_exact_grid_constants_first_and_no_condition_or_extra_parameters():
    policies = gate.policy_grid()
    assert len(policies) == len({p["id"] for p in policies}) == 17
    assert {(p["alpha_observed"], p["alpha_missing"]) for p in policies} == {
        (a, b) for a in gate.ENDPOINTS for b in gate.ENDPOINTS} | {(1., 1.)}
    assert [p["alpha_observed"] for p in policies[:5]] == [0., .25, .5, .75, 1.]
    assert [p["complexity"] for p in policies] == [0] * 5 + [1] * 12
    assert sum(p["unchanged"] for p in policies) == 1
    for p in policies:
        gate.validate_policy(p)
        gate.validate_policy({**p, "id": "candidate"})
    for changed in ({"parent_confidence": .8}, {"alpha_missing": .6}, {"condition": "clean"}):
        with pytest.raises(ValueError):
            gate.validate_policy({**policies[0], **changed})


def test_raw_mask_and_centered_nearest_window_are_half_local():
    ball = np.zeros((39, 2), np.float32)
    ball[0, 0] = np.nan; ball[20:30] = np.nan; ball[-1, 1] = np.inf
    raw = {"ball": ball, "players": None, "team": None, "pitch": None}
    mask = gate.missing_mask(raw)
    expected = np.r_[True, np.zeros(19, bool), np.ones(10, bool), np.zeros(8, bool), True]
    np.testing.assert_array_equal(mask, expected)
    reference = np.convolve(np.pad(mask.astype(float), (12, 12), mode="edge"), np.ones(25) / 25, mode="valid")
    np.testing.assert_allclose(gate.missing_fraction(mask), reference, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(gate.missing_fraction(np.zeros(7, bool)), 0)
    np.testing.assert_array_equal(gate.missing_fraction(np.ones(7, bool)), 1)
    with pytest.raises(ValueError):
        gate.missing_mask({**raw, "condition": "clean"})
    for bad in (np.zeros(4), np.zeros((4, 1), bool), np.array([], bool)):
        with pytest.raises(ValueError):
            gate.missing_fraction(bad)


@pytest.mark.parametrize("alpha", [0., .25, .5, .75, 1.])
def test_constant_policy_exact_base_parity_including_x_and_zero(alpha):
    rng = np.random.default_rng(178)
    parent = rng.integers(0, 4, 131, dtype=np.int8)
    logits = rng.normal(size=(131, 5)).astype(np.float32)
    mask = rng.integers(0, 2, 131).astype(bool)
    result = gate.compose(parent, logits, mask, policy(alpha, alpha))
    np.testing.assert_array_equal(result, gate.base.compose(parent, logits, alpha, .98))
    if alpha == 0:
        assert result is not parent
    with pytest.raises(ValueError):
        gate.compose(parent, logits, mask[:-1], policy(alpha, alpha))


def test_dynamic_gate_interpolation_and_exact_unchanged_typed_hazards(monkeypatch):
    rng = np.random.default_rng(121)
    n = 71
    parent = rng.integers(0, 4, n, dtype=np.int8)
    logits = rng.normal(size=(n, 5)).astype(np.float32)
    mask = np.zeros(n, bool); mask[21:51] = True
    captured = []
    monkeypatch.setattr(gate.base.native, "decode", lambda nodes, hazard: captured.append((nodes.copy(), hazard.copy())) or parent.copy())
    gate.compose(parent, logits, mask, policy(.25, .75))
    fraction = np.convolve(np.pad(mask.astype(float), (12, 12), mode="edge"), np.ones(25) / 25, mode="valid")
    alpha = (.25 * (1 - fraction) + .75 * fraction).astype(np.float32)[:, None]
    probabilities = gate.base.emissions_api.emissions(torch.tensor(logits[:, 0]), torch.tensor(logits[:, 1])).exp().numpy()
    prior = np.full_like(probabilities, .01)
    known = parent < 3
    prior[np.flatnonzero(known), parent[known]] = .98
    prior[~known] = 1 / 3
    np.testing.assert_array_equal(captured[0][1], logits[:, 2:])
    np.testing.assert_allclose(captured[0][0], np.log(np.clip((1 - alpha) * prior + alpha * probabilities, 1e-8, 1)), atol=3e-7, rtol=0)
    assert alpha[0, 0] == .25 and alpha[35, 0] == .75
    # Interleaving another mask cannot route subsequent work through stale state.
    gate.compose(parent, logits, ~mask, policy(.25, .75))
    gate.compose(parent, logits, mask, policy(.25, .75))
    np.testing.assert_array_equal(captured[0][0], captured[2][0])
    assert not np.array_equal(captured[0][0], captured[1][0])


def test_dynamic_team_symmetry_and_rejection_of_bad_inputs():
    rng = np.random.default_rng(249)
    parent = rng.integers(0, 4, 137, dtype=np.int8)
    logits = rng.normal(size=(137, 5)).astype(np.float32)
    mask = rng.integers(0, 2, 137).astype(bool)
    mapping = np.array([1, 0, 2, 3], np.int8)
    swapped = logits.copy(); swapped[:, 0] *= -1
    for p in gate.policy_grid()[5:]:
        np.testing.assert_array_equal(gate.compose(mapping[parent], swapped, mask, p),
                                      mapping[gate.compose(parent, logits, mask, p)])
    bad_logits = logits.copy(); bad_logits[0, 0] = np.nan
    with pytest.raises(ValueError):
        gate.compose(parent, bad_logits, mask, policy(0., 0.))
    with pytest.raises(ValueError):
        gate.compose(parent.astype(float), logits, mask, policy(.25, .5))


def test_evaluator_prefers_constant_over_dynamic_on_exact_metric_ties(monkeypatch):
    evaluation = gate.study.evaluation
    monkeypatch.setattr(evaluation, "official_metrics", lambda: None)
    monkeypatch.setattr(evaluation, "score_full", lambda *args: ({"spell_f1_mean": .7, "macro_f1": .9}, {}))
    dynamic, constant = policy(.25, .5), policy(.75, .75)
    candidates = [dynamic, constant]
    cases = {(h, c): {p["id"]: np.zeros(4, np.int8) for p in candidates}
             for h in evaluation.HALVES for c in evaluation.CONDITIONS}
    result = evaluation.select_policy(cases, {h: np.zeros(4, np.int8) for h in evaluation.HALVES}, candidates)
    assert result["selected_id"] == constant["id"]


@pytest.fixture
def caches(tmp_path, monkeypatch):
    base, study = gate.base, gate.study
    campaign = tmp_path / "runs/wqq-090-20261031"
    source = campaign / "studies/layer1-v70-prior"
    folder = campaign / "studies/gate"
    source.mkdir(parents=True); folder.mkdir(parents=True)
    for module in (gate, base, gate.runtime):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(study, "CAMPAIGN", campaign)
    monkeypatch.setattr(study.parents, "CAMPAIGN", campaign)
    data = tmp_path / "data"; data.mkdir()
    monkeypatch.setattr(base, "DATA", data)
    scripts = tmp_path / "scripts"; scripts.mkdir()
    parent_source = scripts / "parents.py"; parent_source.write_text("synthetic parent source")
    monkeypatch.setattr(study.parents, "__file__", str(parent_source))
    model_protocol = tmp_path / "model-protocol.json"; model_protocol.write_text('{"sources": {}}')
    base.atomic_json(source / "protocol.json", {"candidate_id": source.name, "source_hashes": {},
                     "recipe": {"model_family": "v70", "model_protocol": base.pin(model_protocol), "parent": "synthetic"}})
    (source / "oof-seal.json").write_text("{}")
    original_pin = base.pin(source / "protocol.json")
    root_protocol = {"kind": gate.KIND, "candidate_id": folder.name, "source_hashes": {},
                     "recipe": {"model_family": "v70", "model_protocol": base.pin(model_protocol),
                                "policies": gate.policy_grid(), "parent_confidence": gate.Q,
                                "gate": gate.gate_spec(), "composition": "raw-missing25-native-three-state-v1",
                                "conditions": ["clean", "corrupted"], "corruption_seed": 100},
                     "source_study": base.relative(source), "source_protocol": original_pin,
                     "source_oof_seal": base.pin(source / "oof-seal.json"), "cache_index": {}}
    raw_calls = []
    def raw_fixture(match, half, condition):
        raw_calls.append((match, half, condition))
        path = data / f"{match}-{half}.npz"
        with np.load(path, allow_pickle=False) as archive:
            raw = {k: archive[k].copy() for k in archive.files}
        if condition == "corrupted":
            raw["ball"][8:25] = np.nan
        return raw, base.pin(path)
    monkeypatch.setattr(base, "raw_fixture", raw_fixture)
    def forbidden(*args, **kwargs):
        pytest.fail("Outer/WPY gate attempted model or parent inference")
    monkeypatch.setattr(base, "predict_logits", forbidden)
    monkeypatch.setattr(study.parents, "get_parent", forbidden)
    def create(which="J03WMX", match="J03WPY", half="firstHalf", condition="clean"):
        key = gate.view_key(which, match, half, condition)
        name = key.split("/", 1)[1]
        raw_path = data / f"{match}-{half}.npz"
        if not raw_path.exists():
            np.savez(raw_path, ball=np.zeros((33, 2), np.float32), players=np.zeros((33, 2, 2), np.float32),
                     team=np.array([0, 1]), pitch=np.array([105, 68]))
        parent_path = campaign / "parents-v2" / which / f"{name}.npz"
        parent_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(parent_path, parent=np.resize(np.array([0, 0, 1, 2, 3], np.int8), 33))
        parent_meta = {"which": which, "match": match, "half": half, "condition": condition,
                       "artifact": base.pin(parent_path), "source": base.pin(parent_source), "nomination": None,
                       "fit_matches": base.cohort(which), "calibration_matches": ["J03WPY"],
                       "ancestry_verified": True, "upstream_evidence": [base.pin(parent_source)]}
        base.atomic_json(parent_path.with_suffix(".json"), parent_meta)
        weight_path = tmp_path / f"weight-{which}.bin"
        weight_path.write_bytes(b"synthetic weights")
        freeze_path = tmp_path / f"freeze-{which}.json"
        base.atomic_json(freeze_path, {"fit_matches": base.cohort(which), "model": base.pin(weight_path)})
        logit_path = source / "logits" / which / f"{name}.npz"
        logit_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(logit_path, logits=np.random.default_rng(15).normal(size=(33, 5)).astype(np.float32))
        logit_meta = {"which": which, "match": match, "half": half, "condition": condition, "family": "v70",
                      "output": base.pin(logit_path), "raw_input": base.pin(raw_path), "sources": {},
                      "corruption_seed": 100 if condition == "corrupted" else None, "reference_reads": [],
                      "model_bundle": {"learned_ancestors": [{"freeze": base.pin(freeze_path)}]}}
        base.atomic_json(logit_path.with_suffix(".json"), logit_meta)
        view_path = source / "views" / which / f"{name}.npz"
        view_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(view_path, candidate=np.zeros(33, np.int8))
        base.atomic_json(view_path.with_suffix(".json"), {"identity": {"which": which, "match": match,
                         "half": half, "condition": condition, "protocol": original_pin}, "artifact": base.pin(view_path),
                         "inputs": [base.pin(logit_path.with_suffix(".json")), base.pin(parent_path.with_suffix(".json"))],
                         "reference_reads": []})
        entry = gate._cache_entry(source, original_pin, which, match, half, condition)
        root_protocol["cache_index"][key] = entry
        base.atomic_json(folder / "protocol.json", root_protocol)
        return entry
    def produce(condition="clean", match="J03WPY", policies=None):
        return gate.produce_view(folder, "J03WMX", match, "firstHalf", condition, policies or gate.policy_grid())
    monkeypatch.setattr(gate, "source_pins", lambda: {})
    return SimpleNamespace(root=tmp_path, source=source, folder=folder, protocol=root_protocol,
                           create=create, produce=produce, raw_calls=raw_calls)


def test_cached_views_never_infer_and_resume_is_exact(caches):
    caches.create(condition="clean"); caches.create(condition="corrupted")
    clean = caches.produce("clean")
    corrupt = caches.produce("corrupted")
    resumed = caches.produce("clean")
    assert clean == resumed
    assert len(caches.raw_calls) == 2
    assert clean["identity"]["mask"] != corrupt["identity"]["mask"]
    assert clean["reference_reads"] == [] and len(clean["arrays"]) == 17
    with np.load(gate.base.checked(clean["artifact"]), allow_pickle=False) as values:
        parent = np.load(gate.base.checked(caches.protocol["cache_index"]["J03WMX/J03WPY-firstHalf-clean"]["parent"]))["parent"]
        np.testing.assert_array_equal(values["o000_m000"], parent)


def test_generated_ancestry_and_arrays_seal_and_select_with_unchanged_evaluator(caches):
    for half in gate.base.HALVES:
        for condition in ("clean", "corrupted"):
            caches.create(half=half, condition=condition)
    rows, nodes = [], {}
    for half in gate.base.HALVES:
        for condition in ("clean", "corrupted"):
            meta = gate.produce_view(caches.folder, "J03WMX", "J03WPY", half, condition, gate.policy_grid())
            rows.append(gate.study.row_for(meta))
            gate.study.add_nodes(nodes, meta["ancestry_nodes"])
    path = caches.folder / "calibration-seal.json"
    gate.study.evaluation.create_seal(path, run_id=caches.folder.name, stage="calibration", rows=rows,
        policies=gate.policy_grid(), protocol=gate.base.pin(caches.folder / "protocol.json"),
        ancestry_nodes=nodes, producer_held="J03WMX", root=caches.root)
    truth = np.resize(np.array([0, 0, 1, 2, 3], np.int8), 33)
    selected = gate.study.evaluation.select_policy_from_seal(path, root=caches.root,
        truth_loader=lambda match, half: truth.copy())
    assert selected["selected_id"] == "o000_m000"
    assert selected["policy_shared_across_conditions"] is True
    assert selected["producer_held"] == "J03WMX"
    assert {row["match"] for row in selected["reference_reads"]} == {"J03WPY"}


def test_changed_source_raw_identity_fails_on_output_resume(caches):
    entry = caches.create()
    caches.produce()
    path = gate.base.checked(entry["raw_input"])
    with path.open("ab") as out:
        out.write(b"changed physical observations")
    with pytest.raises(ValueError, match="raw fixture"):
        caches.produce()


def test_protocol_rejects_unregistered_window_before_accessing_observations(caches):
    caches.create()
    changed = copy.deepcopy(caches.protocol)
    changed["recipe"]["gate"]["window_frames"] = 125
    gate.base.atomic_json(caches.folder / "protocol.json", changed)
    with pytest.raises(ValueError, match="Unregistered missing25"):
        caches.produce()
    assert caches.raw_calls == []


@pytest.mark.parametrize("field,value", [("condition", "corrupted"), ("which", "J03WN1"), ("corruption_seed", 100)])
def test_misrouted_cached_logit_receipt_is_rejected_before_any_inference(caches, field, value):
    entry = caches.create()
    path = gate.base.checked(entry["logit_receipt"])
    metadata = gate.base.read_json(path); metadata[field] = value
    gate.base.atomic_json(path, metadata)
    with pytest.raises(ValueError):
        caches.produce()
    assert caches.raw_calls == []


def test_mask_resume_cannot_borrow_another_conditions_gate_state(caches):
    caches.create(condition="clean"); caches.create(condition="corrupted")
    clean = caches.produce("clean")
    receipt = caches.folder / "masks/J03WMX/J03WPY-firstHalf-corrupted.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(gate.base.checked(clean["identity"]["mask_receipt"]).read_text())
    with pytest.raises(ValueError, match="mask resume identity"):
        caches.produce("corrupted")


def test_composition_resume_rejects_changed_policies_or_output(caches):
    caches.create()
    metadata = caches.produce()
    with pytest.raises(ValueError, match="composition resume identity"):
        caches.produce(policies=[policy(.25, .5)])
    with gate.base.checked(metadata["artifact"]).open("ab") as out:
        out.write(b"changed output")
    with pytest.raises(ValueError, match="hash-bound"):
        caches.produce()


def test_unindexed_or_missing_cache_does_not_fall_back_to_model(caches):
    caches.create()
    with pytest.raises(PermissionError, match="cache index"):
        caches.produce("corrupted")
    entry = caches.protocol["cache_index"]["J03WMX/J03WPY-firstHalf-clean"]
    gate.base.checked(entry["logit_receipt"]).unlink()
    with pytest.raises(FileNotFoundError):
        caches.produce()


def test_register_pins_all_32_existing_views_and_cannot_replace_protocol(caches, monkeypatch):
    for which in gate.base.TRAIN:
        for match in (which, "J03WPY"):
            for half in gate.base.HALVES:
                for condition in ("clean", "corrupted"):
                    caches.create(which, match, half, condition)
    monkeypatch.setattr(gate.study.evaluation, "load_seal", lambda *args, **kwargs: (
        {"run_id": caches.source.name, "protocol": gate.base.pin(caches.source / "protocol.json")}, None))
    destination = caches.folder.with_name("registered")
    protocol = gate.register(destination, caches.source)
    assert len(protocol["cache_index"]) == 32 and len(protocol["recipe"]["policies"]) == 17
    assert caches.raw_calls == []
    assert gate.register(destination, caches.source) == protocol
    path = destination / "protocol.json"
    changed = copy.deepcopy(protocol); changed["recipe"]["parent_confidence"] = .8
    gate.base.atomic_json(path, changed)
    with pytest.raises(FileExistsError):
        gate.register(destination, caches.source)


def test_full_and_reserved_roles_require_exact_admission_before_any_inference(caches):
    caches.create()
    for match in ("J03WPY", "J03WQQ"):
        with pytest.raises(PermissionError):
            gate.produce_view(caches.folder, "full", match, "firstHalf", "clean", gate.policy_grid())
    for match in ("J03WR9", "challenge"):
        with pytest.raises(ValueError):
            gate.produce_view(caches.folder, "full", match, "firstHalf", "clean", gate.policy_grid())
    permit = caches.root / "permit.json"
    permit.write_text(json.dumps({"candidate_id": "another-study", "protocol": {}, "selected_recipe": {}}))
    with pytest.raises(PermissionError, match="exact study"):
        gate.produce_view(caches.folder, "full", "J03WPY", "firstHalf", "clean", gate.policy_grid(), permit)
    with pytest.raises(PermissionError, match="exact study"):
        gate.full_audit(caches.folder, permit)
    assert not (caches.folder / "full-admission.json").exists()
    assert caches.raw_calls == []


def test_exact_canonical_recipe_check_precedes_full_pipeline(caches, monkeypatch):
    caches.create()
    permit = caches.root / "permit.json"
    nomination = {"candidate_id": caches.folder.name, "protocol": gate.base.pin(caches.folder / "protocol.json"),
                  "selected_recipe": copy.deepcopy(caches.protocol["recipe"])}
    nomination["selected_recipe"]["policies"] = [policy(.25, .25)]
    permit.write_text(json.dumps(nomination))
    with pytest.raises(PermissionError, match="exact study"):
        gate.full_audit(caches.folder, permit)
    assert caches.raw_calls == []


def test_full_success_path_admits_before_entering_frozen_driver(caches, monkeypatch):
    caches.create()
    permit = caches.root / "permit.json"; permit.write_text("{}")
    events = []
    monkeypatch.setattr(gate.runtime, "preflight_full", lambda folder, p: events.append(("admitted", folder, p)) or {"kind": "synthetic"})
    monkeypatch.setattr(gate.base, "pin", lambda path: {"path": str(path), "sha256": "synthetic"})
    monkeypatch.setattr(gate, "protocol_for", lambda folder: caches.protocol)
    def full(folder, p):
        assert events[0] == ("admitted", folder, p)
        assert gate.study.produce_view is gate.produce_view
        assert (folder / "full-admission.json").exists()
        return "synthetic-finished"
    monkeypatch.setattr(gate.study, "full_audit", full)
    assert gate.full_audit(caches.folder, permit) == "synthetic-finished"


def test_scoped_producer_adapter_restores_on_failure():
    original = gate.study.produce_view
    with pytest.raises(RuntimeError):
        with gate.installed():
            assert gate.study.produce_view is gate.produce_view
            raise RuntimeError("synthetic failure")
    assert gate.study.produce_view is original


def test_frozen_source_manifest_includes_new_driver_and_tests():
    pins = gate.source_pins()
    for path in ("scripts/wqq_campaign_gate.py", "tests/test_wqq_campaign_gate.py"):
        assert pins[path] == gate.base.sha256(gate.ROOT / path)
