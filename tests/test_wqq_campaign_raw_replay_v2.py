"""Small synthetic replays only; no historical arrays, weights or references."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from scripts import wqq_campaign_raw_replay_v2 as replay


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def frozen_function(path, name, globals_):
    """Execute the actual frozen function body, with IO supplied in memory."""
    tree = ast.parse((replay.ROOT / path).read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    env = dict(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), env)
    return env[name]


def raw(n=600, gaps=True):
    rng = np.random.default_rng(7)
    value = {"ball": rng.normal(0, 3, (n, 2)).astype(np.float32),
             "players": rng.normal(0, 15, (n, 22, 2)).astype(np.float32),
             "team": np.repeat(np.array([0, 1], np.int8), 11),
             "pitch": np.array([105, 68], np.float32)}
    if gaps:
        value["ball"][100:240] = np.nan
    return value


class Classifier:
    def predict_proba(self, x):
        p = replay.expit(np.clip(x[:, 0] * .03 + x[:, 1] * .02, -1, 1))
        return np.c_[1-p, p]


class Quality:
    def predict(self, x):
        return np.clip(.45 + .15 * np.tanh(x[:, 0] * .01), 0, 1)


class Credit:
    def predict(self, x):
        return .4 + .2 * np.tanh(x[:, 1] * .01)


class Verifier:
    def predict(self, x):
        return np.zeros(len(x))


def config():
    return {"credit": {"alpha": .06918452395436446},
            "stacked": {"alpha": .908259749672592, "temperature": .634477952668133},
            "region": {"observed_alpha": .2745799900544321},
            "verifier": {"slope": 0., "intercept": .00370370375889319},
            "bridge": {"owner_alpha": .75, "beta": .05, "temperature": 2.},
            "legacy": {"alpha": .5, "change_offset": 0.}, "status": copy.deepcopy(replay.SELECTION)}


def test_pure_api_rejects_nonraw_fields_and_bad_geometry():
    v = raw(20)
    replay.validate_raw(v)
    for key in ("truth", "condition", "match", "parent", "logits"):
        with pytest.raises(ValueError):
            replay.validate_raw({**v, key: 0})
    for field, bad in (("team", np.zeros(22)), ("pitch", [0, 68]), ("players", np.zeros((19, 22, 2)))):
        with pytest.raises(ValueError):
            replay.validate_raw({**v, field: bad})
    assert list(inspect.signature(replay.ReplayBundle.predict).parameters) == ["self", "raw"]
    assert list(inspect.signature(replay.ReplayBundle.parent_from_raw).parameters) == ["self", "raw"]
    with pytest.raises(ValueError):
        replay.validate_fixture("J03WQQ", "clean")


def test_canonical747_features_match_frozen_source_exactly():
    v = raw(180)
    rng = np.random.default_rng(3)
    signals = rng.normal(size=(180, 13)).astype(np.float32)
    owners = {f"owner_{i}": rng.normal(size=180).astype(np.float32) for i in range(3)}
    component = frozen_function("scripts/gap_spell_evidence.py", "component",
                                {"np": np, "logit": replay.logit, "expit": replay.expit})
    expected = component(signals, owners)
    np.testing.assert_array_equal(replay.canonical_signals(signals, owners), expected)
    arcs = np.array([[0, 50, 0], [40, 150, 1], [150, 180, 0]])
    oracle = np.c_[replay.RawIntervals(v).features(arcs), replay.graph.segment_features(expected, arcs)].astype(np.float32)
    np.testing.assert_array_equal(replay.EvidenceIntervals(v, expected).features(arcs), oracle)


def test_rebuilt_v20_context_matches_frozen_v9_v10_v11_v18_functions(tmp_path):
    v = raw()
    n = len(v["ball"])
    rng = np.random.default_rng(19)
    labels = np.repeat(np.array([0, 1, 0, 1, 0, 1], np.int8), 100)
    labels[400:430] = 2
    signals = rng.normal(size=(n, 13)).astype(np.float32)
    signals[:, 0] = np.sin(np.arange(n) / 35).astype(np.float32)
    support = replay.uniform_filter1d((~np.isfinite(v["ball"]).all(1)).astype(float), 25, mode="nearest")
    p = {"predicted": labels, "signals": signals, "status_mask": labels.copy(),
         "segments": replay.gs.spells(labels), "eligible_mask": (support > 0) & (labels < 2),
         **{f"owner_{i}": rng.normal(size=n).astype(np.float32) for i in range(3)}}
    models = {"quality": Quality(), "credit": Credit(), "stacked": Classifier(), "verifier": Verifier()}
    cal = config()
    actual = replay.rebuild_context(v, p, models, cal)
    cc = cal["credit"]
    stacked_cal = {**cal["stacked"], "model": "stacked"}
    region_cal = {**cal["region"], "model": "stacked"}
    def frozen_read(path):
        return cc if "credit" in str(path) else region_cal if "region" in str(path) else stacked_cal
    b = NS(raw_half=lambda _: v, original_quality=lambda _: models["quality"],
           fitted=lambda _: (models["credit"], {}), dense=replay.dense, old_quality=replay.old_quality)
    e = NS(RUN=Path("credit"), parent_wpy=lambda _: tmp_path,
           component=replay.canonical_signals, EvidenceIntervals=replay.EvidenceIntervals)
    s = NS(RUN=Path("stacked"), read=frozen_read, fitted=lambda _: (models["stacked"], {"model": "stacked"}),
           predict_owner=replay.predict_owner, owner=replay.stacked, decode_graph=replay.retained.decode_graph, TRAIN=[])
    context = frozen_function("second_zero/selector_full_parents.py", "context",
                              {"np": np, "b": b, "e": e, "s": s, "gs": replay.gs,
                               "region": NS(RUN=Path("region")), "expit": replay.expit, "logit": replay.logit,
                               "uniform_filter1d": replay.uniform_filter1d, "ROOT": tmp_path,
                               "load_archive": lambda *args: p})
    oracle = context("full", "J03WPY", "firstHalf")
    edits = replay.selector.candidates(oracle["baseline"], oracle["alternatives"], oracle["pool"], oracle["pool_quality"])
    features, proxy, _ = replay.selector.feature_matrix(v, oracle["baseline_signals"], oracle["rich_signals"],
                                                       oracle["baseline"], edits, models["quality"])
    ids = replay.verifier.select_boundaries(edits, proxy)
    boundary = replay.verifier.apply(oracle["baseline"], [edits[i] for i in ids], np.arange(len(ids)))
    # Original v11 writes only a synthetic in-memory-derived case here.
    archive = tmp_path / "J03WPY-firstHalf-corrupted.npz"
    np.savez_compressed(archive, **p)
    index_rows = frozen_function("scripts/metric_decoder_v11.py", "index_rows", {"np": np})
    make_case = frozen_function("scripts/metric_decoder_v11.py", "make_case",
        {"np": np, "time": NS(monotonic=lambda: 0.), "b": b, "e": e, "gs": replay.gs,
         "boundary_parent": lambda *args: boundary, "read": lambda _: cc, "fast": replay.fast,
         "index_rows": index_rows, "previous": NS(load_case=lambda *args: ({"baseline": oracle["baseline"]}, None, None)),
         "prior": NS(producer_proof=lambda _: {}), "RUN": tmp_path / "metric", "pin": lambda p: str(p)})
    meta = make_case("full", "J03WPY", "firstHalf")
    with np.load(meta["cache"]) as z:
        metric_case = {k: z[k].copy() for k in z.files}
    nested_context = frozen_function("scripts/nested_context_calibration_v18.py", "context",
                                     {"d": NS(load_case=lambda *args: (metric_case, {})), "gs": replay.gs})
    expected, _ = nested_context("full", "J03WPY", "firstHalf")
    for key in ("baseline", "arcs", "gap_arc_mask", "gap_frame_mask", "unary", "credit"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    np.testing.assert_array_equal(actual["boundary_parent"], metric_case["baseline"])


def test_parent_composition_matches_frozen_v20_and_exact_zero_bypass():
    n = 120
    parent = np.repeat(np.array([0, 1, 0], np.int8), 40)
    mask = np.ones(n, bool)
    arcs = replay.gs.pool(parent, mask, np.array([[0, 60, 0], [60, 120, 1]]))
    rng = np.random.default_rng(18)
    v = {"baseline": parent, "arcs": arcs, "gap_arc_mask": np.ones(len(arcs), bool),
         "gap_frame_mask": mask, "unary": rng.normal(size=n).astype(np.float32),
         "credit": np.linspace(.2, .8, len(arcs))}
    signals = rng.normal(size=(n, 13)).astype(np.float32)
    student, boundary = rng.normal(size=(2, n)).astype(np.float32)
    support = np.linspace(0, 1, n).astype(np.float32)
    cfg = config()["bridge"]
    r = NS(b=NS(old_quality=replay.old_quality, original_quality=lambda _: Quality()),
           read=lambda _: {"alpha": .06918452395436446}, d=NS(e=NS(RUN=Path("dummy"))))
    original = frozen_function("scripts/gap_bridge_v20.py", "compose",
                               {"np": np, "expit": replay.expit, "logit": replay.logit,
                                "r": r, "d": NS(fast=replay.fast), "gs": replay.gs,
                                "endpoint_credit": replay.endpoint_credit})
    actual = replay.compose_parent(v, signals, student, boundary, support, cfg, Quality(), .06918452395436446)
    np.testing.assert_array_equal(actual, original(v, signals, (student, boundary), support, cfg, "full"))
    zero = {**cfg, "owner_alpha": 0., "beta": 0.}
    result = replay.compose_parent(v, None, None, None, None, zero, None, 0)
    np.testing.assert_array_equal(result, parent)
    assert result is not parent


def test_observed_parent_identity_and_shared_status_policy(monkeypatch):
    v = raw(32, gaps=False)
    parent = np.resize(np.array([0, 1, 2, 3], np.int8), 32)
    logits = np.random.default_rng(11).normal(size=(32, 5)).astype(np.float32)
    pipeline = NS(predict=lambda raw: {"predicted": parent.copy()})
    bundle = replay.ReplayBundle(pipeline, {}, config(), NS())
    def native(given, given_bundle):
        assert given is v
        assert torch.get_num_threads() == 1
        return logits.copy()
    monkeypatch.setattr(replay.base, "infer_raw", native)
    before = replay.raw_identity(v)
    arrays, trace = bundle.predict(v)
    np.testing.assert_array_equal(arrays["parent"], parent)
    np.testing.assert_array_equal(arrays["status_mass"], replay.status.compose(parent, logits, replay.SELECTION))
    assert replay.raw_identity(v) == before
    assert trace["observed_only_parent_identity"] is True
    with pytest.raises(TypeError):
        bundle.predict(v, condition="clean")


def test_legacy_bypass_cannot_leak_into_shared_policy_or_be_misrouted():
    v = raw(24, gaps=False)
    arrays = {"parent": np.zeros(24, np.int8), "logits": np.ones((24, 5), np.float32)}
    fixture = {"match": "J03WPY", "half": "firstHalf", "condition": "clean",
               "observation_identity": replay.raw_identity(v), "legacy_mode": "clean-bypass"}
    np.testing.assert_array_equal(replay.legacy_fixture_prediction(arrays, v, fixture), arrays["parent"])
    corrupt = {**fixture, "condition": "corrupted", "legacy_mode": "joint-alpha050"}
    np.testing.assert_array_equal(replay.legacy_fixture_prediction(arrays, v, corrupt),
                                  replay.base.native.compose(arrays["parent"], arrays["logits"], .5, 0.))
    for wrong in ({"match": "J03WQQ"}, {"legacy_mode": "joint-alpha050"}, {"observation_identity": {}}, {"half": "bad"}):
        with pytest.raises(ValueError):
            replay.legacy_fixture_prediction(arrays, v, {**fixture, **wrong})
    changed = copy.deepcopy(v); changed["ball"][0, 0] += 1
    with pytest.raises(ValueError):
        replay.legacy_fixture_prediction(arrays, changed, fixture)


def test_inference_boundary_blocks_truth_audit_cached_outputs_even_if_allowlisted(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    names = ("runs/parent/J03WPY-firstHalf.npz", "runs/study/oof-audit.json", "runs/held/J03WQQ.json",
             "raw/J03WPY-firstHalf.truth.npy", "raw/J03WR9.npz", "evaluation/gt/target.json")
    for name in names:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"forbidden")
    good = tmp_path / "weights/model.pt"; good.parent.mkdir(); good.write_bytes(b"weights")
    with replay.ReadBoundary([good, *(tmp_path / name for name in names)]):
        assert good.read_bytes() == b"weights"
        for name in names:
            with pytest.raises(PermissionError):
                (tmp_path / name).read_bytes()
    # A compare grant permits only explicitly listed saved arrays; truth stays shut.
    cached = tmp_path / names[0]
    with replay.ReadBoundary([cached, tmp_path / names[3]], comparison=True):
        assert cached.read_bytes() == b"forbidden"
        with pytest.raises(PermissionError):
            (tmp_path / names[3]).read_bytes()


def test_commit_interruption_and_resume_never_rewrites_committed_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    work = tmp_path / "replay"
    identity = {"input": "pinned", "sources": "v1"}
    arrays = {"parent": np.zeros(3, np.int8)}
    original = replay.immutable_json
    def interrupted(*args):
        raise RuntimeError("interrupted before receipt commit")
    monkeypatch.setattr(replay, "immutable_json", interrupted)
    with pytest.raises(RuntimeError):
        replay.commit_view(work, "view", identity, arrays, {})
    orphan = next((work / "fresh").glob("*.npz"))
    before_orphan = orphan.read_bytes()
    monkeypatch.setattr(replay, "immutable_json", original)
    meta = replay.commit_view(work, "view", identity, arrays, {})
    payload = replay.checked(meta["output"])
    assert payload != orphan and orphan.read_bytes() == before_orphan
    receipt = work / "fresh/view.json"
    snapshots = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (payload, receipt)}
    replay.commit_view(work, "view", identity, {"parent": np.ones(3, np.int8)}, {})
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshots} == snapshots
    with pytest.raises(ValueError):
        replay.commit_view(work, "view", {**identity, "input": "different"}, arrays, {})
    payload.write_bytes(b"mutated")
    with pytest.raises(ValueError):
        replay.committed_view(receipt, identity)


def test_frozen_policy_and_same_path_mutation_rejected(tmp_path, monkeypatch):
    replay.validate_semantics(config())
    for key, change in (("bridge", {"owner_alpha": .5}), ("legacy", {"alpha": .25}),
                        ("status", {"alpha": .5}), ("verifier", {"slope": .1})):
        cal = config(); cal[key].update(change)
        with pytest.raises(ValueError):
            replay.validate_semantics(cal)
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    p = tmp_path / "model.pt"; p.write_bytes(b"original")
    item = replay.pin(p); p.write_bytes(b"changed!")
    with pytest.raises(ValueError):
        replay.checked(item)


def pilot(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(replay, "source_closure", lambda: {})
    monkeypatch.setattr(replay, "runtime_identity", lambda: {})
    monkeypatch.setattr(replay, "repair_provenance", lambda: {"synthetic": True})
    monkeypatch.setattr(replay, "compiled_loaders", lambda sources: {})
    monkeypatch.setattr(replay, "verify_kernel_inventory", lambda spec: None)
    work = tmp_path / "work"; work.mkdir()
    fixtures = {}
    expected = tmp_path / "expected.npz"
    np.savez(expected, parent=np.zeros(200, np.int8))
    for h in replay.HALVES:
        p = tmp_path / f"private/improvement/external/prepared/J03WPY-{h}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **raw(200, gaps=False))
        for c in replay.CONDITIONS:
            fixtures[f"J03WPY-{h}-{c}"] = {"half": h, "condition": c, "raw_input": replay.pin(p),
                "expected": {"parent": {"artifact": replay.pin(expected), "array": "parent"}}}
    p = {"kind": replay.KIND, "runtime": {}, "sources": {}, "repair": {"synthetic": True},
         "spec": {"calibration": config(), "runtime_pins": [], "compiled_kernels": {"loaders": {}}},
         "fixtures": fixtures, "comparison_receipts": []}
    replay.immutable_json(work / "protocol.json", p)
    return work, p, expected


def test_compare_requires_all_fresh_commits_before_expected_array_open(tmp_path, monkeypatch):
    work, protocol, expected = pilot(tmp_path, monkeypatch)
    loaded = []
    original = np.load
    def spy(path, *args, **kwargs):
        loaded.append(Path(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(replay.np, "load", spy)
    with pytest.raises(FileNotFoundError):
        replay.compare(work)
    assert expected not in loaded
    protocol_pin = replay.pin(work / "protocol.json")
    for key, row in protocol["fixtures"].items():
        r = replay.fixture_raw(row)
        identity = replay.view_identity(protocol_pin, row, r)
        replay.commit_view(work, key, identity, {"parent": np.zeros(200, np.int8)}, {})
    views = {k: replay.pin(work / "fresh" / (k + ".json")) for k in protocol["fixtures"]}
    replay.immutable_json(work / "fresh-seal.json", {"protocol": protocol_pin, "views": views})
    result = replay.compare(work)
    assert result["all_arrays_bit_exact"] is True and expected in loaded
    assert result["reference_reads"] == [] and result["metric_computation"] is False


def test_compare_rejects_raw_identity_change_before_expected_read(tmp_path, monkeypatch):
    work, protocol, expected = pilot(tmp_path, monkeypatch)
    pp = replay.pin(work / "protocol.json")
    for key, row in protocol["fixtures"].items():
        identity = replay.view_identity(pp, row, replay.fixture_raw(row))
        replay.commit_view(work, key, {**identity, "corruption_seed": -1}, {"parent": np.zeros(200, np.int8)}, {})
    replay.immutable_json(work / "fresh-seal.json", {"protocol": pp, "views": {
        k: replay.pin(work / "fresh" / (k + ".json")) for k in protocol["fixtures"]}})
    expected.unlink()
    with pytest.raises(ValueError, match="identity"):
        replay.compare(work)


def test_fixture_loader_pins_raw_and_rejects_extra_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    p = tmp_path / "private/improvement/external/prepared/J03WPY-firstHalf.npz"
    p.parent.mkdir(parents=True)
    np.savez(p, **raw(200, False))
    row = {"half": "firstHalf", "condition": "corrupted", "raw_input": replay.pin(p)}
    expected = replay.retained.corruption(raw(200, False), 100)
    actual = replay.fixture_raw(row)
    assert replay.raw_identity(actual) == replay.raw_identity(expected)
    np.savez(p, **raw(200, False), truth=np.zeros(200))
    with pytest.raises(ValueError, match="Changed pinned"):
        replay.fixture_raw(row)
    row["raw_input"] = replay.pin(p)
    with pytest.raises(ValueError, match="extra fields"):
        replay.fixture_raw(row)
    row["raw_input"]["path"] = row["raw_input"]["path"].replace("J03WPY", "J03WQQ")
    with pytest.raises(PermissionError):
        replay.fixture_raw(row)


def test_predict_stage_commits_all_views_without_reading_expected_then_resumes(tmp_path, monkeypatch):
    work, protocol, expected = pilot(tmp_path, monkeypatch)
    seen = []
    class Bundle:
        def predict(self, v):
            # Even knowing the exact comparison location grants no input access.
            with pytest.raises(PermissionError):
                expected.read_bytes()
            seen.append(replay.raw_identity(v))
            n = len(v["ball"])
            return {"parent": np.zeros(n, np.int8), "logits": np.zeros((n, 5), np.float32),
                    "status_mass": np.zeros(n, np.int8)}, {}
    monkeypatch.setattr(replay, "load_bundle", lambda spec: Bundle())
    replay.predict_wpy(work)
    assert len(seen) == 4 and (work / "fresh-seal.json").exists()
    snapshots = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (work / "fresh").iterdir()}
    replay.predict_wpy(work)
    assert len(seen) == 4
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshots} == snapshots
    assert replay.compare(work)["all_arrays_bit_exact"] is True


def test_prepare_only_builds_manifest_and_never_loads_array_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    data = tmp_path / "private/improvement/external/prepared"
    data.mkdir(parents=True)
    fixtures = {}
    halves = []
    for h in replay.HALVES:
        p = data / f"J03WPY-{h}.npz"; p.write_bytes(b"not-an-array-payload")
        item = replay.pin(p)
        halves.append({"name": f"J03WPY-{h}", "input_sha256": item["sha256"]})
        for c in replay.CONDITIONS:
            fixtures[f"J03WPY-{h}-{c}"] = {"raw_input": item}
    (data / "J03WPY.json").write_text(json.dumps({"match": "J03WPY", "halves": halves}))
    monkeypatch.setattr(replay, "gather_spec", lambda: {"calibration": config()})
    monkeypatch.setattr(replay, "comparison_inventory", lambda spec: (fixtures, []))
    monkeypatch.setattr(replay, "source_closure", lambda: {"source.py": "pinned"})
    monkeypatch.setattr(replay, "runtime_identity", lambda: {})
    monkeypatch.setattr(replay, "repair_provenance", lambda: {"synthetic": True})
    def forbidden_load(*args, **kwargs):
        pytest.fail("prepare tried to parse an array")
    monkeypatch.setattr(replay.np, "load", forbidden_load)
    result = replay.prepare(tmp_path / "work")
    assert result["comparison_payloads_opened"] == []
    assert result["raw_preparation_receipt"] == replay.pin(data / "J03WPY.json")


@pytest.mark.parametrize("bits", [(0x00000000, 0x80000000), (0x7FC00001, 0x7FC00002)])
def test_bit_exact_rejects_signed_zero_and_different_nan_payloads(bits):
    value = np.array([bits[0]], np.uint32).view(np.float32)
    expected = np.array([bits[1]], np.uint32).view(np.float32)
    result = replay.compare_arrays(value, expected)
    assert result["bit_exact"] is False
    assert result["actual"]["sha256"] != result["expected"]["sha256"]
    assert result["value_equal"] is True
    assert result["different_values"] == 0
    identical = replay.compare_arrays(value, value.copy())
    assert identical["bit_exact"] is True
    assert identical["value_equal"] is True
    assert identical["different_values"] == 0


def test_byte_comparator_keeps_numeric_differences_shape_and_dtype_separate():
    value = np.array([0., 2.], np.float32)
    result = replay.compare_arrays(value, np.array([0., 3.], np.float32))
    assert result["bit_exact"] is False and result["value_equal"] is False
    assert result["different_values"] == 1
    dtype = replay.compare_arrays(value, value.astype(np.float64))
    assert dtype["bit_exact"] is False and dtype["value_equal"] is True
    assert dtype["different_values"] == 0
    shape = replay.compare_arrays(value, value.reshape(1, 2))
    assert shape["bit_exact"] is False and shape["value_equal"] is False
    assert "different_values" not in shape


def test_versioned_repair_preserves_v1_and_all_scientific_function_bodies():
    proof = replay.repair_provenance()
    assert proof["scientific_change"] is False
    assert replay.KIND.endswith("v2") and replay.DEFAULT_WORK.name == "raw-replay-v2"
    original = ast.parse(replay.checked(proof["prior_source"]).read_text())
    updated = ast.parse(Path(replay.__file__).read_text())
    for name in ("validate_raw", "array_pin", "compare_arrays", "raw_identity", "old_quality",
                 "canonical_signals", "EvidenceIntervals", "rebuild_context", "compose_parent",
                 "ReplayBundle", "legacy_fixture_prediction", "fixture_raw", "commit_view"):
        left = next(n for n in original.body if getattr(n, "name", None) == name)
        right = next(n for n in updated.body if getattr(n, "name", None) == name)
        assert ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False), name


def test_reachable_compiled_loader_inventory_contains_exact_two_kernels():
    inventory = replay.kernel_inventory()
    assert set(inventory["loaders"]) == {"second_zero/sprint_decoder.py", "second_zero/fast_spell_graph.py"}
    assert inventory["automatic_compilation"] is False
    assert {k["source"]["path"] for k in inventory["kernels"]} == {
        "runs/two-hour-sprint/decoder/decoder.c", "runs/metric-decoder-v11/kernel/graph.c"}
    assert {k["binary"]["path"] for k in inventory["kernels"]} == {
        "runs/two-hour-sprint/decoder/decoder.dylib", "runs/metric-decoder-v11/kernel/graph.dylib"}
    for kernel in inventory["kernels"]:
        assert kernel["source"]["sha256"] == kernel["embedded_source_sha256"]


def test_real_baseline_graph_path_cold_loads_only_pinned_kernels_without_compilation(monkeypatch):
    inventory = replay.kernel_inventory()
    paths = [replay.checked(k[name]) for k in inventory["kernels"] for name in ("source", "binary")]
    snapshots = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    def no_compile(*args, **kwargs):
        pytest.fail("Inference attempted to compile a frozen kernel")
    monkeypatch.setattr(replay.sprint_decoder.subprocess, "run", no_compile)
    monkeypatch.setattr(replay.sprint_decoder, "_LIB", None)
    monkeypatch.setattr(replay.fast, "_LIB", None)
    n = 160
    signals = np.random.default_rng(27).normal(size=(n, 13)).astype(np.float32)
    parent = np.resize(np.array([0, 0, 1, 1], np.int8), n)
    old_allowlist = [p for p in paths if p.parent == replay.fast.BUILD]
    with replay.ReadBoundary(old_allowlist):
        with pytest.raises(PermissionError, match="decoder.c"):
            replay.retained.decode_graph(signals, parent, Quality())
    with replay.ReadBoundary(paths):
        fresh, arcs, _ = replay.retained.decode_graph(signals, parent, Quality())
        final = replay.fast.Graph(signals[:, 0], parent, replay.gs.spells(parent)).decode(
            np.ones(len(replay.gs.spells(parent))), .4, .01)
    assert fresh.shape == final.shape == parent.shape and len(arcs) > 0
    assert replay.sprint_decoder._LIB is not None and replay.fast._LIB is not None
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths} == snapshots


def test_kernel_source_mismatch_missing_binary_and_unregistered_loader_fail_closed(tmp_path, monkeypatch):
    module_file = tmp_path / "kernel.py"; module_file.write_text("# fixture\n")
    module = NS(__file__=str(module_file), BUILD=tmp_path, SOURCE="frozen source\n")
    source = tmp_path / "tiny.c"; source.write_text(module.SOURCE)
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="precompiled"):
        replay.kernel_files(module, "tiny")
    binary = tmp_path / "tiny.dylib"; binary.write_bytes(b"synthetic-library")
    expected = replay.kernel_files(module, "tiny")
    assert expected["source"]["sha256"] == expected["embedded_source_sha256"]
    source.write_text("different source\n")
    with pytest.raises(ValueError, match="frozen Python SOURCE"):
        replay.kernel_files(module, "tiny")
    source.write_text(module.SOURCE)
    binary.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="precompiled"):
        replay.kernel_files(module, "tiny")


def test_verified_kernel_allowlist_is_explicit_and_complete(monkeypatch):
    inventory = replay.kernel_inventory()
    pins = [k[field] for k in inventory["kernels"] for field in ("source", "binary")]
    spec = {"compiled_kernels": inventory, "runtime_pins": pins}
    replay.verify_kernel_inventory(spec)
    with pytest.raises(ValueError, match="allowlist"):
        replay.verify_kernel_inventory({**spec, "runtime_pins": pins[1:]})
    changed = copy.deepcopy(inventory); changed["kernels"][0]["binary"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        replay.verify_kernel_inventory({**spec, "compiled_kernels": changed})
    monkeypatch.setattr(replay, "compiled_loaders", lambda sources: {"unknown/kernel.py": []})
    with pytest.raises(ValueError, match="uninventoried"):
        replay.kernel_inventory({})
