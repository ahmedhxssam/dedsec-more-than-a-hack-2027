"""WPY-only raw replay of frozen v20, legacy v70 and shared status-mass.

Inference receives four observation arrays and pinned fitted artifacts. Saved
predictions are comparison targets only, opened by a different CLI stage after
fresh output receipts commit. No fitting, scoring, WQQ or challenge route exists.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from second_zero import fast_spell_graph as fast
from second_zero import full_pipeline_selector as selector
from second_zero import gap_spell as gs
from second_zero import gap_spell_dense as dense
from second_zero import selected_edit_verifier as verifier
from second_zero import sprint_decoder
from second_zero import stacked_gap_owner as stacked
from second_zero.full_baseline_pipeline import ExplicitRawPipeline, gap_candidate, retained
from second_zero.full_baseline_sources import scientific_sources
from second_zero.gap_change_evidence import endpoint_credit
from second_zero.gap_spell_model import RawIntervals
from scripts import sprint_decoder_segments_geometry_v2 as graph
from scripts import wqq_campaign_baseline_ancestry as ancestry
from scripts import wqq_campaign_model as base
from scripts import wqq_campaign_status_mass_v1 as status
from scripts.global_team_context_probe import owner as predict_owner

KIND = "wqq-campaign-strict-raw-replay-v2"
DEFAULT_WORK = ROOT / "runs/wqq-090-20261031/raw-replay-v2"
STUDY = ROOT / "runs/wqq-090-20261031/studies/layer5-v70-parent-status-mass"
BASELINE = ROOT / "runs/full-baseline-gap-v1"
HALVES, CONDITIONS = ("firstHalf", "secondHalf"), ("clean", "corrupted")
RAW_NAMES = frozenset(("ball", "players", "team", "pitch"))
EXPECTED_SELECTION_SHA = "44240781b9285734c4a446cb3db3250dc7c282a6d73c742f34b2e9246f11ddec"
SELECTION = {"alpha": .25, "complexity": 1, "id": "parentD_a025",
             "parent_confidence": .98, "status_mode": "parent_D", "unchanged": False}
HISTORICAL_SOURCES = (
    "scripts/gap_spell_evidence.py", "second_zero/selector_full_parents.py",
    "scripts/full_pipeline_selector_v9.py", "scripts/selected_edit_verifier_v10.py",
    "scripts/metric_decoder_v11.py", "scripts/nested_context_calibration_v18.py",
    "scripts/gap_bridge_v20.py", "scripts/hierarchical_joint_v70.py",
    "scripts/joint_corrupted_v58.py", "scripts/wqq_campaign_raw_replay_v2.py",
    "scripts/prepare_external_football.py",
    "tests/test_wqq_campaign_raw_replay_v2.py",
)
REPAIR = {
    "kind": "raw-replay-v2-explicit-runtime-kernel-closure",
    "reason": "v1 omitted sprint decoder.c/decoder.dylib from the inference read allowlist",
    "scientific_change": False,
    "prior_source": {"path": "scripts/wqq_campaign_raw_replay_v1.py",
                     "sha256": "7e1724cfa3a331ed9cc91ba714feb7d42396b71237d26373b7c445a2a5d28af6"},
    "prior_tests": {"path": "tests/test_wqq_campaign_raw_replay_v1.py",
                    "sha256": "1258c621867375fad662355095492446354262131e1d6d7a63464aaddc44eab1"},
    "prior_protocol": {"path": "runs/wqq-090-20261031/raw-replay-v1/protocol.json",
                       "sha256": "be61e786a9c8d89206522ac8e42955b639e7ab855bb6a6b384c10bfb3c274468"},
    "failed_job_spec": {"path": "runs/wqq-090-20261031/job-specs/raw-replay-v1-pilot.json",
                        "sha256": "5361d472a65a98f141f85ac82023b6b00c28271079ead00759363dde04dd3359"},
    "failed_stdout": {"path": "runs/wqq-090-20261031/jobs/raw-replay-v1-pilot/0b894cce00884124b29d9e75d1a8697c/stdout.log",
                      "sha256": "4a736503aa20e0a393feaed515f9bc9a72148af2a557eef9b1b259bf452466a6"},
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def pin(path):
    return {"path": relative(path), "sha256": digest(path)}


def checked(item):
    p = (ROOT / item["path"]).resolve()
    if not p.is_relative_to(ROOT) or digest(p) != item["sha256"]:
        raise ValueError("Changed pinned artifact: " + item["path"])
    return p


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def immutable_json(path, value):
    """Never replace committed receipts; an abandoned unique temp is harmless."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise FileExistsError("Committed receipt differs: " + str(path))
        return
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    # Hard-link publication is atomic and fails instead of replacing a receipt.
    os.link(temp, path)
    temp.unlink()


def runtime_identity():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: importlib.metadata.version(n) for n in
                         ("numpy", "scipy", "torch", "scikit-learn", "threadpoolctl")},
            "baseline_torch_threads": "owners/status3; GRU/joint2",
            "parent_torch_threads": 2, "parent_BLAS_threads": 2,
            "native_torch_threads": 1, "native_BLAS_threads": 1,
            "corruption_seed": 100}


def source_closure():
    """Static local import closure, including source-only historical recipes."""
    queue = [*map(Path, HISTORICAL_SOURCES), *scientific_sources()]
    seen = set()
    while queue:
        p = Path(queue.pop())
        p = p if p.is_absolute() else ROOT / p
        p = p.resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [node.module + "." + x.name for x in node.names]
            for name in names:
                if name.split(".")[0] in ("scripts", "second_zero", "vendor"):
                    queue.append(ROOT / (name.replace(".", "/") + ".py"))
    return {relative(p): digest(p) for p in sorted(seen)}


def repair_provenance():
    for item in REPAIR.values():
        if isinstance(item, dict) and "path" in item:
            checked(item)
    return REPAIR


def compiled_loaders(sources):
    """Inventory local shared-library loaders in the complete pinned closure."""
    result = {}
    for source in sources:
        tree = ast.parse((ROOT / source).read_text())
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                if name.endswith((".CDLL", ".LoadLibrary", ".load_library", ".dlopen")):
                    calls.append({"line": node.lineno, "call": name})
        if calls:
            result[source] = calls
    return result


def kernel_files(module, stem):
    """Verify existing compiled files; never call legacy auto-build helpers."""
    source, binary = module.BUILD / (stem + ".c"), module.BUILD / (stem + ".dylib")
    if not source.is_file() or source.read_text() != module.SOURCE:
        raise ValueError("Compiled source does not equal its frozen Python SOURCE: " + str(source))
    if not binary.is_file() or binary.stat().st_size == 0:
        raise FileNotFoundError("Required precompiled kernel is absent: " + str(binary))
    return {"module": pin(Path(module.__file__)), "source": pin(source), "binary": pin(binary),
            "embedded_source_sha256": hashlib.sha256(module.SOURCE.encode()).hexdigest()}


def kernel_inventory(sources=None):
    kernels = [kernel_files(sprint_decoder, "decoder"), kernel_files(fast, "graph")]
    loaders = compiled_loaders(source_closure() if sources is None else sources)
    expected = {k["module"]["path"] for k in kernels}
    if set(loaders) != expected:
        raise ValueError("An uninventoried local compiled loader is reachable")
    return {"loaders": loaders, "kernels": kernels, "automatic_compilation": False}


def verify_kernel_inventory(spec):
    expected = spec["compiled_kernels"]
    actual = [kernel_files(sprint_decoder, "decoder"), kernel_files(fast, "graph")]
    if expected["kernels"] != actual or expected["automatic_compilation"] is not False:
        raise ValueError("Runtime compiled kernel identity changed")
    runtime = {p["path"]: p for p in spec["runtime_pins"]}
    for kernel in actual:
        if any(runtime.get(kernel[k]["path"]) != kernel[k] for k in ("source", "binary")):
            raise ValueError("Verified kernel is missing from the explicit runtime allowlist")


def forbidden(path):
    s = str(path)
    return (any(x in s for x in ("J03WQQ", "J03WR9", ".truth", ".actor", "/evaluation/gt/", "/runs/main/"))
            or "audit" in Path(s).name.lower() or Path(s).name.lower() in ("verdict.json", "best.json"))


class ReadBoundary:
    """Read allowlist for inference; compare separately grants sealed targets."""
    def __init__(self, reads, write_root=None, *, comparison=False):
        self.reads = {Path(p).resolve() for p in reads}
        self.write_root = None if write_root is None else Path(write_root).resolve()
        self.active = False
        self.opened = set()
        self.comparison = comparison

    def __call__(self, event, args):
        if not self.active:
            return
        if event in ("subprocess.Popen", "os.system"):
            raise PermissionError("Replay cannot launch subprocesses")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if forbidden(p):
            raise PermissionError("Reference/audit/reserved input is closed")
        if writing:
            if self.write_root is None or not p.is_relative_to(self.write_root):
                raise PermissionError("Replay writes only its new work directory")
            return
        if (not self.comparison and p.suffix in (".npz", ".npy")
                and not (p.parent == ROOT / "private/improvement/external/prepared"
                         and p.name in {f"J03WPY-{h}.npz" for h in HALVES})):
            raise PermissionError("Saved prediction/feature arrays are closed during inference")
        if p in self.reads:
            self.opened.add(p)
            return
        # Already-imported packages sometimes load extension libraries lazily.
        runtime = any(p.is_relative_to(ROOT / x) for x in
                      (".venv", "private/improvement/runtime", "private/improvement/vision-env"))
        system = any(str(p).startswith(x) for x in ("/System/", "/usr/", "/Library/", "/opt/"))
        if (runtime or system) and p.suffix in (".py", ".pyc", ".so", ".dylib"):
            return
        raise PermissionError("Unlisted read (including prediction caches): " + str(p))

    def __enter__(self):
        sys.addaudithook(self)
        self.active = True
        return self

    def __exit__(self, *_):
        self.active = False


def validate_raw(raw):
    if set(raw) != RAW_NAMES:
        raise ValueError("Exactly four raw observation fields are required")
    ball, players, team, pitch = (np.asarray(raw[n]) for n in ("ball", "players", "team", "pitch"))
    if (ball.ndim != 2 or ball.shape[1] != 2 or not len(ball)
            or players.ndim != 3 or players.shape[0] != len(ball) or players.shape[2] != 2
            or team.shape != (players.shape[1],) or set(np.unique(team)) != {0, 1}
            or pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0)
            or any(a.dtype.kind not in "fiu" for a in (ball, players, team, pitch))):
        raise ValueError("Invalid raw tracking geometry")


def array_pin(a):
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def compare_arrays(value, expected):
    """Byte identity and numerical equality are distinct replay claims."""
    actual_pin, expected_pin = array_pin(value), array_pin(expected)
    same_shape = value.shape == expected.shape
    result = {"bit_exact": actual_pin == expected_pin,
              "value_equal": bool(same_shape and np.array_equal(value, expected, equal_nan=True)),
              "actual": actual_pin, "expected": expected_pin}
    if same_shape:
        different = value != expected
        # Paired NaNs are numerically equivalent here, regardless of payload.
        different &= ~(np.isnan(value) & np.isnan(expected))
        result["different_values"] = int(np.count_nonzero(different))
    return result


def raw_identity(raw):
    validate_raw(raw)
    return {k: array_pin(raw[k]) for k in sorted(RAW_NAMES)}


def old_quality(model, signals, arcs):
    out = np.empty(len(arcs))
    for s in range(0, len(arcs), 8192):
        out[s:s+8192] = model.predict(graph.segment_features(signals, arcs[s:s+8192]))
    return np.clip(out, 0, 1)


def canonical_signals(signals, owners):
    x = signals.copy()
    z = np.column_stack([owners["owner_" + str(i)] for i in range(3)]).astype(np.float32)
    x[:, 0] = logit(np.clip(expit(z).mean(1), 1e-10, 1-1e-10))
    return x


class EvidenceIntervals(RawIntervals):
    def __init__(self, raw, signals):
        super().__init__(raw)
        self.signals = signals

    def features(self, arcs):
        return np.c_[super().features(arcs), graph.segment_features(self.signals, arcs)].astype(np.float32)


def rebuild_context(raw, p, models, calibration):
    """Exact selector_full_parents -> v10 boundary -> v11 -> v18 context."""
    quality, credit = models["quality"], models["credit"]
    cc, cal, rc = (calibration[k] for k in ("credit", "stacked", "region"))
    mask = p["eligible_mask"]
    original = dense.proposals(p["predicted"], mask, p["segments"])
    canonical_x = canonical_signals(p["signals"], p)
    qr = EvidenceIntervals(raw, canonical_x).predict(credit, original)
    qo = old_quality(quality, p["signals"], original)
    primary = gs.decode(p["predicted"], mask, original, (1-cc["alpha"])*qo+cc["alpha"]*qr,
                        .4, p["signals"][:, 0], .01)
    new = expit(predict_owner(models["stacked"], stacked.features(raw, canonical_x))/cal["temperature"])
    support = uniform_filter1d((~np.isfinite(raw["ball"]).all(1)).astype(np.float64), 25, mode="nearest")
    alternatives, all_arcs, rich = [], [original], None
    for name in ("fixed025", "calibrated", "region"):
        weight = (.25*support if name == "fixed025" else cal["alpha"]*support if name == "calibrated"
                  else cal["alpha"]*support+rc["observed_alpha"]*(1-support))
        signals = p["signals"].copy()
        signals[:, 0] = logit(np.clip((1-weight)*expit(signals[:, 0])+weight*new, 1e-7, 1-1e-7)).astype(np.float32)
        preliminary, aa, _ = retained.decode_graph(signals, p["status_mask"], quality)
        if name == "region" and rc["observed_alpha"] > 0:
            extra = dense.proposals(preliminary, mask, aa)
            aa = gs.pool(preliminary, preliminary < 2, np.r_[aa, extra, gs.spells(primary)])
            mutable = preliminary < 2
        else:
            preliminary[~mask] = primary[~mask]
            aa = dense.proposals(preliminary, mask, aa)
            mutable = mask
        oq = old_quality(quality, signals, aa)
        nq = EvidenceIntervals(raw, canonical_signals(signals, p)).predict(credit, aa)
        alternatives.append(gs.decode(preliminary, mutable, aa, (1-cc["alpha"])*oq+cc["alpha"]*nq,
                                      .4, signals[:, 0], .01))
        all_arcs.append(aa)
        if name == "region":
            rich = signals
    pool = gs.pool(primary, primary < 2, np.concatenate(all_arcs))
    edits = selector.candidates(primary, alternatives, pool, old_quality(quality, p["signals"], pool))
    features, proxy, _ = selector.feature_matrix(raw, p["signals"], rich, primary, edits, quality)
    selected = verifier.select_boundaries(edits, proxy)
    chosen = [edits[i] for i in selected]
    x = verifier.features(features, proxy, edits, selected)
    gains = models["verifier"].predict(x)
    if not np.isfinite(gains).all():
        raise FloatingPointError("Nonfinite frozen verifier gain")
    keep = verifier.accepted(gains, calibration["verifier"])
    # WPY's v11 path applied every proposal; WQQ's path applied verified ones.
    # The frozen full calibrator is constant positive, so these are identical.
    if not np.array_equal(keep, np.arange(len(chosen))):
        raise ValueError("Full verifier no longer matches the frozen WPY/full equivalence")
    boundary = verifier.apply(primary, chosen, keep)
    arcs = gs.pool(boundary, boundary < 2, np.r_[original, p["segments"], gs.spells(boundary)])
    credit_values = EvidenceIntervals(raw, canonical_x).predict(credit, arcs)
    # v18 replaces v11's boundary parent and recomputes feasibility, but retains
    # the boundary-derived arc inventory. Do not rebuild this pool on primary.
    return {"baseline": primary, "boundary_parent": boundary, "arcs": arcs,
            "gap_arc_mask": gs.feasible(arcs, primary, mask), "gap_frame_mask": mask,
            "unary": p["signals"][:, 0], "credit": credit_values}


def compose_parent(v, signals, student, boundary, support, cfg, quality, credit_alpha):
    a, b, t = cfg["owner_alpha"], cfg["beta"], cfg["temperature"]
    if not (a or b):
        return v["baseline"].copy()
    weight = a*support
    z = logit(np.clip((1-weight)*expit(v["unary"])+weight*expit(student/t), 1e-7, 1-1e-7)).astype(np.float32)
    take = v["gap_arc_mask"]
    arcs = v["arcs"][take]
    x = signals.copy(); x[:, 0] = z
    qo = old_quality(quality, x, arcs)
    q = (1-credit_alpha)*qo+credit_alpha*v["credit"][take]
    q += endpoint_credit(arcs, v["baseline"], support, boundary, b, 1.)
    out = fast.Graph(z, v["baseline"], arcs).decode(q, .4, .01)
    gs.check_lock(out, v["baseline"], v["gap_frame_mask"])
    return out


class ReplayBundle:
    def __init__(self, pipeline, models, calibration, native_bundle):
        self.pipeline, self.models, self.calibration, self.native_bundle = pipeline, models, calibration, native_bundle

    def parent_from_raw(self, raw):
        validate_raw(raw)
        with threadpool_limits(2):
            p = self.pipeline.predict(raw)
            trace = {"complete_baseline": array_pin(p["predicted"])}
            if np.isfinite(raw["ball"]).all():
                return p["predicted"].copy(), {}, {**trace, "observed_only_parent_identity": True}
            torch.set_num_threads(2)
            z = predict_owner(self.models["global"], RawIntervals(raw).x[:, :, :290])
            p = {**p, **gap_candidate(raw, p, z, self.models["quality"])}
            v = rebuild_context(raw, p, self.models, self.calibration)
            x, prior, support = base.context_api.inputs(raw)
            student, boundary = base.context_api.predict_both(self.native_bundle.context_model, x, prior, len(raw["ball"]))
            out = compose_parent(v, p["signals"], student, boundary, support, self.calibration["bridge"],
                                 self.models["quality"], self.calibration["credit"]["alpha"])
            intermediates = {"credit_parent": v["baseline"], "boundary_parent": v["boundary_parent"],
                             "arcs": v["arcs"], "credit": v["credit"], "gap_frame_mask": v["gap_frame_mask"],
                             "unary": v["unary"], "bridge_owner": student, "bridge_boundary": boundary,
                             "bridge_support": support}
            trace.update({k: array_pin(a) for k, a in intermediates.items()})
            return out, intermediates, trace

    def native_from_raw(self, raw):
        validate_raw(raw)
        with threadpool_limits(1):
            torch.set_num_threads(1)
            return base.infer_raw(raw, self.native_bundle)

    def predict(self, raw):
        """Pure shared policy: deliberately has no fixture/match/condition arg."""
        before = raw_identity(raw)
        parent, intermediates, trace = self.parent_from_raw(raw)
        logits = self.native_from_raw(raw)
        prediction = status.compose(parent, logits, SELECTION)
        if raw_identity(raw) != before:
            raise ValueError("Replay mutated caller observations")
        return {"parent": parent, "logits": logits, "status_mass": prediction, **intermediates}, trace


def legacy_fixture_prediction(arrays, raw, fixture):
    """Historical clean bypass is fixture-bound and outside the shared API."""
    validate_fixture(fixture["half"], fixture["condition"])
    if fixture.get("match") != "J03WPY" or fixture.get("observation_identity") != raw_identity(raw):
        raise ValueError("Historical execution mode is not bound to these observations")
    mode = "clean-bypass" if fixture["condition"] == "clean" else "joint-alpha050"
    if fixture.get("legacy_mode") != mode:
        raise ValueError("Wrong frozen historical execution mode")
    if mode == "clean-bypass":
        if not np.isfinite(raw["ball"]).all():
            raise ValueError("WPY clean replay requires its fully observed original input")
        return arrays["parent"].copy()
    return base.native.compose(arrays["parent"], arrays["logits"], .5, 0.)


def validate_fixture(half, condition):
    if half not in HALVES or condition not in CONDITIONS:
        raise ValueError("Only the four registered WPY views exist")


def validate_semantics(cal):
    if (cal["bridge"] != {"owner_alpha": .75, "beta": .05, "temperature": 2.}
            or cal["legacy"] != {"alpha": .5, "change_offset": 0.}
            or cal["status"] != SELECTION or cal["verifier"]["slope"] != 0
            or cal["verifier"]["intercept"] <= 0):
        raise ValueError("The approved frozen composition changed")


def gather_spec():
    """Metadata/weights only. Never execute old drivers or load their arrays."""
    runtime_pins, proof_pins = {}, {}
    def record(p, expected=None, provenance=False):
        p = Path(p); p = p if p.is_absolute() else ROOT / p
        if forbidden(p):
            raise PermissionError("Closed dependency during prepare")
        item = pin(p)
        if expected is not None and item["sha256"] != expected:
            raise ValueError("Frozen historical dependency changed: " + relative(p))
        (proof_pins if provenance else runtime_pins)[item["path"]] = item
        return item
    def document(p, expected=None, provenance=False):
        return read(ROOT / record(p, expected, provenance)["path"])
    baseline = document(BASELINE / "full-reference-spec.json")
    for item in baseline["owners"] + [v for v in baseline.values() if isinstance(v, dict) and "path" in v]:
        record(item["path"], item["sha256"])
    proof = ancestry.validate_full_baseline()
    for item in proof["evidence"]:
        proof_pins[item["path"]] = item
    model_paths = {"credit": "gap-spell-v3-evidence", "stacked": "stacked-gap-recovery-v7",
                   "verifier": "selected-edit-verifier-v10", "bridge": "gap-bridge-v20",
                   "native": "hierarchical-joint-v70"}
    models, freezes = {}, {}
    def ancestor(item):
        f = document(item["path"], item["sha256"], provenance=True)
        cohort = f.get("fit_matches", f.get("residual_fit_matches"))
        if set(cohort or []) != set(base.TRAIN):
            raise ValueError("Native learned ancestor cohort mismatch")
        record(f["model"]["path"], f["model"]["sha256"], provenance=True)
        for k in ("context_teacher", "weight_initialization"):
            if f.get(k):
                ancestor(f[k]["freeze"])
        return f
    for key, folder in model_paths.items():
        path = ROOT / f"runs/{folder}/models/full/freeze.json"
        f = document(path)
        fit = f.get("fit_matches", f.get("residual_fit_matches", f.get("verifier_fit_matches", f.get("train_matches"))))
        if set(fit or []) != set(base.TRAIN):
            raise ValueError("Full producer is not TRAIN4-only")
        models[key] = record(f["model"]["path"], f["model"]["sha256"])
        freezes[key] = f
        if "protocol" in f:
            item = f["protocol"]
            protocol = document(item["path"], item["sha256"])
        else:
            name = "scorer-protocol.json" if key == "credit" else "protocol.json"
            protocol = document(ROOT / f"runs/{folder}/{name}", f["protocol_sha256"])
        for source, sha in protocol.get("sources", {}).items():
            record(source, sha, provenance=True)
        for a in list(f.get("strict_crossfit_feature_ancestors", {}).values()) + [r["ancestor"] for r in f.get("rows", []) if "ancestor" in r]:
            if a["target_match"] in a["label_matches"] or not set(a["label_matches"]) < set(base.TRAIN):
                raise ValueError("Training feature ancestry is not excluded")
            record(a["manifest"]["path"], a["manifest"]["sha256"], provenance=True)
            for part in a["weight_and_calibration_parts"].values():
                record(part["model"], part["model_sha256"], provenance=True)
                for p, h in part["receipts"].items():
                    record(p, h, provenance=True)
    native = freezes["native"]
    if native["context_teacher"]["model"] != models["bridge"]:
        raise ValueError("Native/context model mismatch")
    ancestor(native["weight_initialization"]["freeze"])
    transfer = document(BASELINE / "full-transfer/protocol.json")
    models["global"] = record(transfer["expert"]["path"], transfer["expert"]["sha256"])
    document(transfer["expert_receipt"]["path"], transfer["expert_receipt"]["sha256"], provenance=True)
    document(transfer["expert_original_protocol"]["path"], transfer["expert_original_protocol"]["sha256"], provenance=True)
    # Full-transfer's own expert-fit receipt is embedded in this frozen protocol.
    if set(transfer["expert_fit"]["fit"]) != set(base.TRAIN) or not transfer["expert_fit"]["WPY_excluded_from_expert_weights"]:
        raise ValueError("Global expert fit cohort mismatch")
    cal = {}
    for key, fields, selected in (
        ("credit", ("alpha",), False), ("stacked", ("alpha", "temperature"), False),
        ("verifier", ("slope", "intercept"), False),
        ("bridge", ("owner_alpha", "beta", "temperature"), True),
        ("native", ("alpha", "change_offset"), True)):
        v = document(ROOT / f"runs/{model_paths[key]}/models/full/calibration.json")
        if "model" in v and v["model"] != models[key]:
            raise ValueError("Calibrator/model mismatch")
        if "calibration_matches" in v and not set(v["calibration_matches"]) <= set(base.TRAIN) | {"J03WPY"}:
            raise ValueError("Calibration role includes a held reference")
        v = v["selected"] if selected else v
        cal["legacy" if key == "native" else key] = {k: v[k] for k in fields}
    region = document("runs/region-gap-recovery-v8/calibration/full.json")
    if region["model"] != models["stacked"]:
        raise ValueError("Region calibration uses another expert")
    cal["region"] = {k: region[k] for k in ("observed_alpha", "gap_alpha", "temperature")}
    selection = document(STUDY / "calibration/full/selection.json", EXPECTED_SELECTION_SHA)
    cal["status"] = selection["selected"]
    validate_semantics(cal)
    kernels = kernel_inventory()
    for kernel in kernels["kernels"]:
        for key in ("source", "binary"):
            item = kernel[key]
            record(item["path"], item["sha256"])
    return {"baseline": baseline, "models": models, "calibration": cal,
            "compiled_kernels": kernels,
            "runtime_pins": list(runtime_pins.values()),
            "ancestry": {"baseline": proof, "extra_pins": list(proof_pins.values()),
                         "native_freeze": native, "fit_matches": list(base.TRAIN),
                         "training_payloads_are_provenance_only": True}}


def comparison_inventory(spec):
    """Read receipts, never the expected prediction payloads, before inference."""
    rows, receipt_pins = {}, {}
    def doc(p):
        item = pin(p); receipt_pins[item["path"]] = item
        return read(p)
    old = doc(ROOT / "runs/hierarchical-joint-v70/calibration-cases/full/seal.json")
    case = doc(ROOT / "runs/metric-decoder-v11/case-seal.json")
    for half in HALVES:
        historical = next(p for p in old["outputs"] if Path(p["path"]).name == half + ".npz")
        if historical["path"] != f"runs/hierarchical-joint-v70/calibration-cases/full/{half}.npz":
            raise PermissionError("Historical target is outside the frozen WPY calibration cases")
        graph_case = case["rows"][f"WPY-full-{half}"]["cache"]
        for condition in CONDITIONS:
            key = f"J03WPY-{half}-{condition}"
            v = doc(STUDY / "views/full" / (key + ".json"))
            ident = v["identity"]
            if any(ident[k] != x for k, x in {"which": "full", "match": "J03WPY", "half": half, "condition": condition}.items()):
                raise ValueError("Saved comparison view is misrouted")
            refs = ident["input_identity"]
            native = doc(checked(refs["logit_receipt"]))
            parent = doc(checked(refs["parent_receipt"]))
            if (native["output"] != refs["logits"] or native["raw_input"] != refs["raw_input"]
                    or parent["artifact"] != refs["parent"] or v["reference_reads"]):
                raise ValueError("Comparison receipts disagree")
            if (native["model_bundle"]["model"] != spec["models"]["native"]
                    or native["model_bundle"]["context_model"] != spec["models"]["bridge"]
                    or native["reference_reads"] or parent["reference_arrays_opened"]):
                raise ValueError("Saved comparison uses another fitted model or read references")
            expected = {"parent": {"artifact": refs["parent"], "array": "parent"},
                        "logits": {"artifact": refs["logits"], "array": "logits"},
                        "status_mass": {"artifact": v["artifact"], "array": SELECTION["id"]},
                        "legacy_v70": {"artifact": historical if condition == "corrupted" else refs["parent"],
                                       "array": "param_2" if condition == "corrupted" else "parent"}}
            if condition == "corrupted":
                for output, name in (("credit_parent", "baseline_07708"), ("boundary_parent", "baseline"),
                                     ("arcs", "arcs"), ("credit", "credit"), ("gap_frame_mask", "gap_frame_mask"), ("unary", "unary")):
                    expected[output] = {"artifact": graph_case, "array": name}
                expected["historical_logits"] = {"artifact": historical, "array": "logits", "actual": "logits"}
            for target in expected.values():
                if "J03WPY" not in target["artifact"]["path"] and target["artifact"] != historical:
                    raise PermissionError("Comparison target is outside WPY")
            rows[key] = {"half": half, "condition": condition, "raw_input": refs["raw_input"], "expected": expected}
    return rows, list(receipt_pins.values())


def prepare(work=DEFAULT_WORK):
    work = Path(work).resolve()
    if (work / "protocol.json").exists():
        return protocol_for(work)
    spec = gather_spec()
    fixtures, comparison_receipts = comparison_inventory(spec)
    raw_pins = {v["raw_input"]["path"]: v["raw_input"] for v in fixtures.values()}
    provenance_path = ROOT / "private/improvement/external/prepared/J03WPY.json"
    provenance = read(provenance_path)
    if provenance["match"] != "J03WPY" or any(
        row["input_sha256"] != raw_pins[f"private/improvement/external/prepared/{row['name']}.npz"]["sha256"]
        for row in provenance["halves"]
    ):
        raise ValueError("WPY raw preparation provenance differs")
    for item in raw_pins.values():
        checked(item)  # Hash only; no raw array loading during prepare.
    result = {"kind": KIND, "scope": "WPY-only; no WQQ route", "spec": spec,
              "repair": repair_provenance(),
              "sources": source_closure(), "runtime": runtime_identity(), "fixtures": fixtures,
              "raw_preparation_receipt": pin(provenance_path),
              "comparison_receipts": comparison_receipts, "reference_reads": [],
              "comparison_payloads_opened": [], "recipe": "complete/raw290/credit747/boundary-context/v20/native70/status-mass"}
    immutable_json(work / "protocol.json", result)
    return result


def protocol_for(work):
    p = read(Path(work) / "protocol.json")
    if p["kind"] != KIND or p["runtime"] != runtime_identity() or p["sources"] != source_closure():
        raise ValueError("Replay source/runtime identity changed")
    if p["repair"] != repair_provenance():
        raise ValueError("Versioned repair provenance changed")
    if p["spec"]["compiled_kernels"]["loaders"] != compiled_loaders(p["sources"]):
        raise ValueError("Compiled loader closure differs from the frozen inventory")
    verify_kernel_inventory(p["spec"])
    validate_semantics(p["spec"]["calibration"])
    if set(p["fixtures"]) != {f"J03WPY-{h}-{c}" for h in HALVES for c in CONDITIONS}:
        raise ValueError("Replay fixture set changed")
    for item in p["spec"]["runtime_pins"]:
        checked(item)
    return p


def load_bundle(spec):
    verify_kernel_inventory(spec)
    pipeline = ExplicitRawPipeline(spec["baseline"])
    models = {"quality": pipeline.quality}
    for key in ("global", "credit", "stacked", "verifier"):
        with checked(spec["models"][key]).open("rb") as f:
            models[key] = pickle.load(f)
    model = base.native.JointRecovery().eval()
    model.load_state_dict(torch.load(checked(spec["models"]["native"]), map_location="cpu", weights_only=True))
    context_model = base.context_api.GapBridge().eval()
    context_model.load_state_dict(torch.load(checked(spec["models"]["bridge"]), map_location="cpu", weights_only=True))
    return ReplayBundle(pipeline, models, spec["calibration"], SimpleNamespace(model=model, context_model=context_model))


def fixture_raw(row):
    validate_fixture(row["half"], row["condition"])
    expected = f"private/improvement/external/prepared/J03WPY-{row['half']}.npz"
    if row["raw_input"]["path"] != expected:
        raise PermissionError("Only the pinned WPY raw half may be loaded")
    with np.load(checked(row["raw_input"]), allow_pickle=False) as z:
        if set(z.files) != RAW_NAMES:
            raise ValueError("Prepared input has extra fields")
        raw = {k: z[k].copy() for k in sorted(RAW_NAMES)}
    if row["condition"] == "corrupted":
        raw = retained.corruption(raw, 100)
    validate_raw(raw)
    return raw


def view_identity(protocol_pin, row, raw):
    return {"protocol": protocol_pin, "raw_input": row["raw_input"], "match": "J03WPY",
            "half": row["half"], "condition": row["condition"], "observation_identity": raw_identity(raw),
            "legacy_mode": "clean-bypass" if row["condition"] == "clean" else "joint-alpha050",
            "corruption_seed": 100 if row["condition"] == "corrupted" else None}


def committed_view(path, identity):
    meta = read(path)
    if meta["identity"] != identity or meta["reference_reads"] != [] or meta["cached_prediction_reads"] != []:
        raise ValueError("Committed replay identity differs")
    checked(meta["output"])
    return meta


def commit_view(work, key, identity, arrays, trace):
    folder = Path(work) / "fresh"
    receipt = folder / (key + ".json")
    if receipt.exists():
        return committed_view(receipt, identity)
    folder.mkdir(parents=True, exist_ok=True)
    payload = folder / (key + "." + uuid.uuid4().hex + ".npz")
    with payload.open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    meta = {"kind": KIND + "-fresh-view", "identity": identity, "output": pin(payload),
            "arrays": {k: array_pin(v) for k, v in arrays.items()}, "trace": trace,
            "reference_reads": [], "cached_prediction_reads": []}
    immutable_json(receipt, meta)
    return meta


def predict_wpy(work=DEFAULT_WORK, half=None, condition=None):
    work = Path(work).resolve()
    p = protocol_for(work)
    protocol_pin = pin(work / "protocol.json")
    rows = [(k, v) for k, v in p["fixtures"].items()
            if (half is None or v["half"] == half) and (condition is None or v["condition"] == condition)]
    if not rows:
        raise ValueError("Empty WPY view selection")
    bundle = None
    for key, row in rows:
        reads = [ROOT / x for x in p["sources"]] + [ROOT / x["path"] for x in p["spec"]["runtime_pins"]]
        reads.append(ROOT / row["raw_input"]["path"])
        # Loading/compute cannot read even our own previously saved predictions.
        with ReadBoundary(reads):
            raw = fixture_raw(row)
            identity = view_identity(protocol_pin, row, raw)
        receipt = work / "fresh" / (key + ".json")
        if receipt.exists():
            committed_view(receipt, identity)
            print("RAW REPLAY RESUME", key, flush=True)
            continue
        with ReadBoundary(reads):
            if bundle is None:
                bundle = load_bundle(p["spec"])
            arrays, trace = bundle.predict(raw)
            arrays["legacy_v70"] = legacy_fixture_prediction(arrays, raw, identity)
        commit_view(work, key, identity, arrays, trace)
        print("RAW REPLAY COMMITTED", key, flush=True)
    if all((work / "fresh" / (k + ".json")).exists() for k in p["fixtures"]):
        receipts = {k: pin(work / "fresh" / (k + ".json")) for k in p["fixtures"]}
        immutable_json(work / "fresh-seal.json", {"kind": KIND + "-fresh-seal", "protocol": protocol_pin,
                                                 "views": receipts, "reference_reads": []})


def compare(work=DEFAULT_WORK):
    """A separate process, only after all four fresh views are committed."""
    work = Path(work).resolve()
    p = protocol_for(work)
    seal = read(work / "fresh-seal.json")
    if seal["protocol"] != pin(work / "protocol.json") or set(seal["views"]) != set(p["fixtures"]):
        raise ValueError("Fresh seal does not bind this complete pilot")
    committed = {}
    # Validate every fresh receipt before granting a single expected-array read.
    for key, item in seal["views"].items():
        meta = read(checked(item)); row = p["fixtures"][key]
        expected_identity = view_identity(seal["protocol"], row, fixture_raw(row))
        committed[key] = committed_view(checked(item), expected_identity)
    for item in p["comparison_receipts"]:
        checked(item)
    reads = [ROOT / v["output"]["path"] for v in committed.values()]
    reads += [ROOT / t["artifact"]["path"] for row in p["fixtures"].values() for t in row["expected"].values()]
    report = {}
    with ReadBoundary(reads, comparison=True):
        for key, meta in committed.items():
            with np.load(checked(meta["output"]), allow_pickle=False) as z:
                actual = {k: z[k].copy() for k in z.files}
            result = {}
            for name, target in p["fixtures"][key]["expected"].items():
                with np.load(checked(target["artifact"]), allow_pickle=False) as z:
                    expected = z[target["array"]]
                    value = actual[target.get("actual", name)]
                    result[name] = compare_arrays(value, expected)
            report[key] = result
    out = {"kind": KIND + "-comparison", "fresh_seal": pin(work / "fresh-seal.json"),
           "all_arrays_bit_exact": all(x["bit_exact"] for r in report.values() for x in r.values()),
           "views": report, "reference_reads": [], "metric_computation": False}
    immutable_json(work / "comparison.json", out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "predict-wpy", "compare"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--half", choices=HALVES)
    parser.add_argument("--condition", choices=CONDITIONS)
    args = parser.parse_args()
    if args.stage != "predict-wpy" and (args.half or args.condition):
        parser.error("View filtering is only available for predict-wpy")
    if args.stage == "prepare":
        p = prepare(args.work)
        print(json.dumps({"protocol": pin(args.work / "protocol.json"), "sources": len(p["sources"]),
                          "runtime_pins": len(p["spec"]["runtime_pins"]), "views": len(p["fixtures"])}))
    elif args.stage == "predict-wpy":
        predict_wpy(args.work, args.half, args.condition)
    else:
        report = compare(args.work)
        print(json.dumps({"all_arrays_bit_exact": report["all_arrays_bit_exact"], "report": pin(args.work / "comparison.json")}))
        if not report["all_arrays_bit_exact"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
