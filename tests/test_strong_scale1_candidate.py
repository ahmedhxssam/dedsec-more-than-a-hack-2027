import importlib.util
from pathlib import Path
import numpy as np
import pytest
from second_zero import strong_scale1_candidate as candidate
from scripts import wqq_campaign_status_mass_v1 as status
from scripts import wqq_campaign_transition_scale_v1 as transition


def test_fixed_composition_and_restart_cost_match_original_math():
    rng = np.random.default_rng(20261115)
    native = rng.normal(size=(81, 5)).astype(np.float32)
    parent = rng.integers(0, 4, 81, dtype=np.int8)
    expected = transition.decode(status.mixed_nodes(parent, native, .5)[0], native[:, 2:], 1.)
    np.testing.assert_array_equal(candidate.compose_logits(parent, native), expected)
    costs = transition.transition_scores(native[:, 2:], 1.)
    np.testing.assert_array_equal(costs[:, 2, 0], -np.logaddexp(0., -native[:, 4].astype(np.float64))-np.log(2.))
    assert candidate.POLICY["apply_source"] is False


def test_raw_route_calls_only_original_parent_native_and_preserves_input():
    n = 31
    raw = dict(ball=np.zeros((n, 2), np.float32), players=np.ones((n, 2, 2), np.float32),
               team=np.array([0, 1], np.int64), pitch=np.array([105.3, 68.]))
    parent = np.zeros(n, np.int8)
    native = np.zeros((n, 5), np.float32)
    calls = []
    class Bundle:
        def parent_from_raw(self, value):
            assert value is raw
            calls.append("parent")
            return parent, {}, {"fixture": "mock"}
        def native_from_raw(self, value):
            assert value is raw
            calls.append("native")
            return native
    before = {k: v.copy() for k, v in raw.items()}
    result, trace = candidate.predict_raw(raw, Bundle())
    assert calls == ["parent", "native"] and trace == {"fixture": "mock"}
    np.testing.assert_array_equal(result["predicted"], candidate.compose_logits(parent, native))
    for key in raw:
        np.testing.assert_array_equal(raw[key], before[key])


def test_inference_guard_excludes_cache_reference_and_external_weight_inputs(tmp_path):
    path = Path(__file__).resolve().parents[1] / "reproduce_strong_scale1.py"
    spec = importlib.util.spec_from_file_location("strong1_guard_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("old.csv", "features.npz", "features.npy", "labels.truth.npy", "model.pt", "model.pkl"):
        with pytest.raises(PermissionError):
            module.reference_guard("open", (str(tmp_path/name), "r", 0))
    module.reference_guard("open", (str(tmp_path/"BAR-ATH_tactical_atd.txt"), "r", 0))
    module.reference_guard("open", (str(tmp_path/"new.csv"), "w", 0))
    staged = (tmp_path/"new-output.csv").resolve()
    module.ALLOWED_OUTPUTS.add(staged)
    module.reference_guard("open", (str(staged), "r", 0))
    module.ALLOWED_OUTPUTS.remove(staged)
