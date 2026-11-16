"""Condition-aware native-model experiments for the bounded WQQ campaign.

The scientific inference API accepts raw observations only. ``condition`` is a
fixture-loader option, never a network input. Historical source and artifacts
are read-only; all new artifacts live below the supplied campaign model root.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "private/improvement/runtime")]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from second_zero import gap_bridge_training as context_api
from second_zero import hierarchical_joint_recovery as native
from second_zero import joint_corrupted_recovery as emissions_api
from second_zero.raw_context_owner import inputs as coarse_inputs
from scripts.train_contextual import corruption

TRAIN = ("J03WMX", "J03WN1", "J03WOH", "J03WOY")
HALVES = ("firstHalf", "secondHalf")
FITS = {**{m: [v for v in TRAIN if v != m] for m in TRAIN}, "full": list(TRAIN)}
DATA = ROOT / "private/improvement/external/prepared"
DEFAULT_WORK = ROOT / "runs/wqq-090-20261031/model"
NATIVE_SEAL = ROOT / "runs/native-joint-v65/native-seal.json"
CONTEXT_ROOT = ROOT / "runs/gap-bridge-v20"
FAMILY_ROOTS = {
    "v70": ROOT / "runs/hierarchical-joint-v70",
    "v74control": ROOT / "runs/empirical-joint-v74/control",
}
SEED, STEPS_PER_MATCH, SAVE_EVERY = 20261204, 1000, 250
PARENT_CONFIDENCES = (.98, .8, .6)
ALPHAS = (0., .25, .5, .75, 1.)
_HASH_DEPTH = 0


class StageAccessGuard:
    """CLI filesystem boundary; integrity hashing never grants parsing access."""
    def __init__(self, stage, fit_matches=(), inference_match=None):
        self.stage = stage
        self.fit_matches = frozenset(fit_matches)
        self.inference_match = inference_match
        self.wqq_authorized = False
        self.parsed_reference_reads = []

    def __call__(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        name = os.fsdecode(args[0])
        if any(token in name for token in ("J03WR9", "/evaluation/gt/", "/runs/main/")):
            raise PermissionError("Reserved WR9 and challenge reference paths remain closed")
        if "J03WQQ" in name and not (self.stage == "predict-logits" and self.wqq_authorized):
            raise PermissionError("WQQ is closed until label-free inference nomination is verified")
        path = Path(name)
        if path.name.endswith(".actor.npy"):
            raise PermissionError("Contact/source label arrays are outside this campaign model")
        if path.name.endswith(".truth.npy"):
            match = path.name.split("-")[0]
            if self.stage == "predict-logits":
                raise PermissionError("Inference cannot open any reference array, including for hashing")
            if _HASH_DEPTH and match in TRAIN:
                return
            if self.stage != "fit" or match not in self.fit_matches:
                raise PermissionError("Reference parsing outside the declared fitting cohort")
            self.parsed_reference_reads.append({"stage": self.stage, "match": match, "path": name})
        if not _HASH_DEPTH and self.stage == "fit" and path.suffix == ".npz":
            if path.parent == DATA and path.name.split("-")[0] not in self.fit_matches:
                raise PermissionError("Raw observations outside the fitting cohort")
            if any(token in name for token in ("/native-joint-v62/native/", "/gap-bridge-v20/training/",
                                               "/raw-context-owner-v17/training/", "/model/cache-clean/")):
                if path.parent.name not in self.fit_matches:
                    raise PermissionError("Feature cache outside the fitting cohort")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    global _HASH_DEPTH
    digest = hashlib.sha256()
    _HASH_DEPTH += 1
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        _HASH_DEPTH -= 1
    return digest.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def pin(path):
    path = Path(path).resolve()
    return {"path": relative(path), "sha256": sha256(path)}


def checked(meta):
    path = (ROOT / meta["path"]).resolve()
    path.relative_to(ROOT)
    if sha256(path) != meta["sha256"]:
        raise ValueError("Changed hash-bound artifact: " + meta["path"])
    return path


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    body = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read_json(path) != value:
            raise FileExistsError("Immutable artifact already exists: " + str(path))
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as out:
        out.write(body)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as out:
        torch.save(value, out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Refusing to overwrite prediction/cache: " + str(path))
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as out:
        np.savez_compressed(out, **arrays)
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def source_pins():
    paths = {Path(__file__).resolve(), ROOT / "scripts/train_contextual.py",
             ROOT / "tests/test_wqq_campaign_model.py"}
    paths.update(Path(mod.__file__).resolve() for name, mod in sys.modules.items()
                 if name.startswith("second_zero.") and getattr(mod, "__file__", None))
    return {relative(p): sha256(p) for p in sorted(paths)}


def verify_sources(sources):
    for path, digest in sources.items():
        checked({"path": path, "sha256": digest})


def policy_grid():
    """Parent identity plus twelve registered mixtures; alpha1 duplicates agree."""
    policies = [{"policy_id": "parent", "alpha": 0., "parent_confidence": .98}]
    for q in PARENT_CONFIDENCES:
        for alpha in ALPHAS[1:]:
            policies.append({"policy_id": f"q{round(q*100):02d}_a{round(alpha*100):03d}",
                             "alpha": alpha, "parent_confidence": q})
    return policies


def compose(parent, logits, alpha, parent_confidence=.98):
    """Shared observation-only policy; exact incumbent for alpha zero.

    This intentionally keeps the v70 transition construction and emission /5
    scaling. Only the declared parent confidence differs from its original
    composition. X is a neutral parent prior, not a fourth physical state.
    """
    p = np.asarray(parent)
    z = np.asarray(logits, np.float32)
    if p.ndim != 1 or p.dtype.kind not in "iu" or not len(p) or not np.isin(p, [0, 1, 2, 3]).all():
        raise ValueError("Nonempty canonical parent labels required")
    if z.shape != (len(p), 5) or not np.isfinite(z).all():
        raise ValueError("Aligned finite native five-channel logits required")
    if alpha not in ALPHAS or parent_confidence not in PARENT_CONFIDENCES:
        raise ValueError("Policy outside the prospectively registered grid")
    if alpha == 0:
        return p.copy()
    tensor = torch.from_numpy(z)
    probabilities = emissions_api.emissions(tensor[:, 0], tensor[:, 1]).exp().numpy()
    prior = np.full_like(probabilities, (1. - parent_confidence) / 2.)
    known = p < 3
    prior[np.flatnonzero(known), p[known].astype(int)] = parent_confidence
    prior[~known] = 1. / 3.
    nodes = np.log(np.clip((1. - alpha) * prior + alpha * probabilities, 1e-8, 1.))
    return native.decode(nodes, z[:, 2:])


def cohort(which):
    if which not in FITS:
        raise ValueError("Unregistered fitting cohort")
    return FITS[which]


def verify_ancestry(freeze, expected, seen=None):
    """Verify every recorded learned initialization and context ancestor."""
    seen = set() if seen is None else seen
    declared = freeze.get("fit_matches", freeze.get("residual_fit_matches"))
    if declared is None or set(declared) != set(expected):
        raise ValueError("Learned ancestor uses the wrong physical matches")
    if "model" in freeze:
        checked(freeze["model"])
    for key in ("weight_initialization", "context_teacher"):
        ancestor = freeze.get(key)
        if not ancestor:
            continue
        if set(ancestor.get("fit_matches", expected)) != set(expected):
            raise ValueError("Learned ancestor cohort mismatch")
        if "model" in ancestor:
            checked(ancestor["model"])
        if "freeze" in ancestor:
            path = checked(ancestor["freeze"])
            if path not in seen:
                seen.add(path)
                verify_ancestry(read_json(path), expected, seen)


@dataclass
class ModelBundle:
    model: torch.nn.Module
    context_model: torch.nn.Module
    family: str
    which: str
    metadata: dict


def load_bundle(family, which, work=DEFAULT_WORK):
    expected = cohort(which)
    if family == "balanced":
        folder = Path(work)
    elif family in ("control", "v74control"):
        folder = FAMILY_ROOTS["v74control"] if which != "full" else Path(work) / "control"
    elif family in FAMILY_ROOTS:
        folder = FAMILY_ROOTS[family]
    else:
        raise ValueError("Unregistered model family")
    path = folder / f"models/{which}/freeze.json"
    freeze = read_json(path)
    verify_ancestry(freeze, expected)
    context_path = CONTEXT_ROOT / f"models/{which}/freeze.json"
    context_freeze = read_json(context_path)
    verify_ancestry(context_freeze, expected)
    if freeze["context_teacher"]["model"] != context_freeze["model"]:
        raise ValueError("Native model and coarse context are not the matched pair")
    model = native.JointRecovery().eval()
    model.load_state_dict(torch.load(checked(freeze["model"]), map_location="cpu", weights_only=True))
    context_model = context_api.GapBridge().eval()
    context_model.load_state_dict(torch.load(checked(context_freeze["model"]), map_location="cpu", weights_only=True))
    meta = {"family": family, "which": which, "fit_matches": expected,
            "model": freeze["model"], "freeze": pin(path),
            "learned_ancestors": [
                {"role": "native", "model": freeze["model"], "freeze": pin(path), "fit_matches": expected},
                {"role": "coarse_context", "model": context_freeze["model"], "freeze": pin(context_path), "fit_matches": expected}],
            "context_model": context_freeze["model"], "context_freeze": pin(context_path)}
    return ModelBundle(model, context_model, family, which, meta)


def context(model, features, n):
    owner, boundary = context_api.predict_both(model, features.astype(np.float32),
                                              np.zeros(features.shape[-1], np.float32), n)
    return np.c_[owner, boundary - np.log(2.)].astype(np.float32)


def infer_raw(raw, bundle):
    """No condition, match identifier, reference or clean teacher is accepted."""
    if set(raw) != {"ball", "players", "team", "pitch"}:
        raise ValueError("Only raw observations are accepted by scientific inference")
    features, _, _ = coarse_inputs(raw)
    coarse = context(bundle.context_model, features, len(raw["ball"]))
    result = native.predict(bundle.model, native.inputs(raw), coarse)
    if result.shape != (len(raw["ball"]), 5) or not np.isfinite(result).all():
        raise FloatingPointError("Invalid complete native inference")
    return result


def canonical_recipe_hash(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_nomination(path, family, work=DEFAULT_WORK):
    nomination = read_json(path)
    if nomination.get("kind") != "wqq-campaign-nomination-v1" or nomination.get("eligible") is not True:
        raise PermissionError("Successful frozen campaign nomination required")
    recipe = nomination["selected_recipe"]
    expected_family = "control" if family == "v74control" else family
    if recipe.get("model_family") != expected_family:
        raise PermissionError("Nomination is for another scientific model family")
    if canonical_recipe_hash(recipe) != nomination["recipe_sha256"]:
        raise PermissionError("Nominated recipe hash mismatch")
    protocol_path = checked(nomination["protocol"])
    protocol = read_json(protocol_path)
    for field in ("sources", "source_hashes"):
        if isinstance(protocol.get(field), dict):
            verify_sources(protocol[field])
    if "model_protocol" in recipe:
        model_protocol = checked(recipe["model_protocol"])
        if model_protocol.resolve() != (Path(work) / "protocol.json").resolve():
            raise PermissionError("Nominated model protocol is not the requested run")
        verify_sources(read_json(model_protocol)["sources"])
    audit = read_json(checked(nomination["oof_audit"]))
    if (audit.get("stage") != "oof" or audit.get("eligible") is not True or
            audit.get("run_id") != nomination["candidate_id"] or
            audit.get("protocol") != nomination["protocol"]):
        raise PermissionError("Nomination does not bind a successful matching OOF audit")
    # The evaluator protocol can be a campaign-wide recipe protocol. Its source
    # manifest, model protocol and audit are separately pinned by the caller.
    return {"nomination": pin(path), "protocol": pin(protocol_path), "recipe": recipe}


def validate_inference_role(which, match, permit=None, family="v70", work=DEFAULT_WORK):
    if match in cohort(which):
        raise PermissionError("Inference cannot audit a model's fitting match")
    if match == "J03WQQ":
        if which != "full" or permit is None:
            raise PermissionError("WQQ inference requires a frozen successful OOF permit")
        validate_nomination(permit, family, work)
    elif match not in TRAIN + ("J03WPY",):
        raise PermissionError("Reserved or unknown match")


def raw_fixture(match, half, condition):
    if match not in TRAIN + ("J03WPY", "J03WQQ") or half not in HALVES:
        raise PermissionError("Unknown or reserved fixture")
    if condition not in ("clean", "corrupted"):
        raise ValueError("Unregistered fixture view")
    path = DATA / f"{match}-{half}.npz"
    with np.load(path, allow_pickle=False) as data:
        raw = {k: data[k].copy() for k in ("ball", "players", "team", "pitch")}
    return (corruption(raw, 100) if condition == "corrupted" else raw), pin(path)


def predict_logits(family, which, match, half, condition, out, work=DEFAULT_WORK, permit=None):
    validate_inference_role(which, match, permit, family, work)
    out = Path(out)
    receipt = out.with_suffix(".json")
    if receipt.exists():
        metadata = read_json(receipt)
        for key, value in {"family": family, "which": which, "match": match, "half": half, "condition": condition}.items():
            if metadata[key] != value:
                raise ValueError("Prediction resume identity mismatch")
        verify_sources(metadata["sources"])
        checked(metadata["output"])
        checked(metadata["raw_input"])
        for entry in metadata["model_bundle"]["learned_ancestors"]:
            checked(entry["model"]); checked(entry["freeze"])
        return metadata
    if out.exists():
        raise FileExistsError("Unsealed output exists; preserve and choose a new output name")
    bundle = load_bundle(family, which, work)
    raw, raw_pin = raw_fixture(match, half, condition)
    started = time.monotonic()
    logits = infer_raw(raw, bundle)
    atomic_npz(out, logits=logits)
    metadata = {"kind": "wqq-campaign-native-logits-v1", "created_utc": now(),
                "family": family, "which": which, "match": match, "half": half,
                "condition": condition, "condition_is_fixture_only": True,
                "output": pin(out), "raw_input": raw_pin, "shape": list(logits.shape),
                "model_bundle": bundle.metadata, "sources": source_pins(),
                "reference_reads": [], "seconds": time.monotonic() - started,
                "corruption_seed": 100 if condition == "corrupted" else None,
                "permit": pin(permit) if permit else None}
    atomic_json(receipt, metadata, immutable=True)
    return metadata


def prepare(work=DEFAULT_WORK):
    work = Path(work)
    work.resolve().relative_to(ROOT / "runs/wqq-090-20261031")
    inputs = {relative(NATIVE_SEAL): sha256(NATIVE_SEAL)}
    original = read_json(NATIVE_SEAL)["rows"]
    if len(original) != 40 or {str(v["view"]) for v in original.values()} != {"100", "271", "failure", "phase0", "phase275"}:
        raise ValueError("Original forty native views do not match the audited source")
    for row in original.values():
        if row["primary_matches"] or not row["coarse_raw_feature_exact"] or row["match"] not in TRAIN:
            raise ValueError("Native view ancestry mismatch")
        for key in ("cache", "coarse"):
            checked(row[key]); inputs[row[key]["path"]] = row[key]["sha256"]
    for which, expected in FITS.items():
        for family in ("v70", "v74control"):
            path = FAMILY_ROOTS[family] / f"models/{which}/freeze.json"
            if not path.exists() and family == "v74control" and which == "full":
                continue
            freeze = read_json(path); verify_ancestry(freeze, expected)
            inputs[relative(path)] = sha256(path)
            inputs[freeze["model"]["path"]] = freeze["model"]["sha256"]
        path = CONTEXT_ROOT / f"models/{which}/freeze.json"
        freeze = read_json(path); verify_ancestry(freeze, expected)
        inputs[relative(path)] = sha256(path)
        inputs[freeze["model"]["path"]] = freeze["model"]["sha256"]
    for match in TRAIN:
        for half in HALVES:
            for extension in (".npz", ".truth.npy"):
                path = DATA / f"{match}-{half}{extension}"
                inputs[relative(path)] = sha256(path)  # Hash only; no references parsed.
    protocol = {"kind": "wqq-campaign-clean-balanced-model-v1", "fits": FITS,
                "seed": SEED, "steps_per_match": STEPS_PER_MATCH,
                "outer_steps": 3000, "full_steps": 4000, "checkpoint_every": SAVE_EVERY,
                "architecture": "Exact v70 typed-hazard native model with matched v20 coarse context",
                "training": {"clean_probability": .5, "original_probability": .5,
                             "batch": 2, "lr": .0001, "weight_decay": .01,
                             "cosine_floor": .1, "gradient_clip": 5., "EMA_decay": .995,
                             "window": 3072, "central": 1024, "checkpoint_selection": "final EMA only"},
                "control": "Reused v74/control outer checkpoints; condition-aware inference and shared calibration applied afresh",
                "families": {"balanced": {"clean_probability": .5},
                             "control": {"clean_probability": 0., "outer_weights": "existing v74/control",
                                         "full_fit": "Matching original40-view continuation from full v70 only if nominated"}},
                "calibration_match": "J03WPY", "policies": policy_grid(),
                "inference": "Only ball/players/team/pitch; no benchmark condition flag, no clean coordinates in corrupted inference",
                "reserved": ["J03WR9", "challenge/public references", "WQQ fitting/calibration"],
                "sources": source_pins(), "inputs": inputs}
    atomic_json(work / "protocol.json", protocol, immutable=True)
    return protocol


def verify_protocol(work=DEFAULT_WORK):
    protocol = read_json(Path(work) / "protocol.json")
    verify_sources(protocol["sources"])
    for path, digest in protocol["inputs"].items():
        checked({"path": path, "sha256": digest})
    return protocol


def cache_clean(work=DEFAULT_WORK):
    work = Path(work)
    verify_protocol(work)
    rows = {}
    for match in TRAIN:
        for half in HALVES:
            key = f"{match}-{half}-clean"
            path = work / f"cache-clean/{match}/{half}.npz"
            receipt = path.with_suffix(".json")
            if receipt.exists():
                row = read_json(receipt); checked(row["cache"]); checked(row["raw_input"])
            else:
                if shutil.disk_usage(work).free < 5 * 1024**3:
                    raise RuntimeError("Clean feature cache paused: less than 5 GiB working reserve")
                raw, raw_pin = raw_fixture(match, half, "clean")
                x = native.inputs(raw)
                coarse, _, _ = coarse_inputs(raw)
                if x.shape != (2, 138, len(raw["ball"])) or coarse.shape != (2, 414, (len(raw["ball"]) + 4) // 5):
                    raise ValueError("Unexpected clean feature dimensions")
                atomic_npz(path, x=x, coarse=coarse.astype(np.float16))
                row = {"cache": pin(path), "coarse": pin(path), "coarse_key": "coarse",
                       "raw_input": raw_pin, "match": match, "half": half, "view": "clean",
                       "shape": list(x.shape), "primary_matches": [], "reference_reads": [],
                       "coarse_raw_feature_exact": True, "native_and_context_same_raw_observations": True,
                       "protocol": pin(work / "protocol.json")}
                atomic_json(receipt, row, immutable=True)
            rows[key] = row
    seal = {"rows": rows, "reference_reads": [], "protocol": pin(work / "protocol.json")}
    atomic_json(work / "clean-seal.json", seal, immutable=True)
    return seal


def balanced_batch(dataset, rng, batch_size=2, clean_probability=.5):
    """Control consumes the same mixture coin as the reused v74 control."""
    if clean_probability not in (0., .5):
        raise ValueError("Registered balanced/control probability required")
    samples = []
    for _ in range(batch_size):
        coin = rng.random()
        group = dataset["clean"] if coin < clean_probability else dataset["original"]
        if not group:
            raise ValueError("Selected observation group is empty")
        samples.append(native.batch(group, rng, batch_size=1))
    return {key: torch.cat([sample[key] for sample in samples], 0) for key in samples[0]}


def checkpoint_state(model, ema, optimizer, rng, step, identity, history, loss_sum, loss_count, elapsed):
    return {"kind": "wqq-campaign-training-checkpoint-v1", "step": step,
            "identity": identity, "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
            "numpy_rng_json": json.dumps(rng.bit_generator.state), "python_rng": random.getstate(),
            "history": history, "loss_sum": loss_sum, "loss_count": loss_count, "elapsed": elapsed}


def restore_state(state, model, ema, optimizer, rng, identity):
    if state.get("kind") != "wqq-campaign-training-checkpoint-v1" or state["identity"] != identity:
        raise ValueError("Checkpoint source/data/model/cohort identity changed")
    model.load_state_dict(state["model"]); ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = json.loads(state["numpy_rng_json"])
    random.setstate(state["python_rng"]); torch.set_rng_state(state["torch_rng"])
    return state


def save_checkpoint(folder, state):
    path = Path(folder) / "checkpoints" / f"step-{state['step']:06d}.pt"
    atomic_torch(path, state)
    receipt = {"kind": state["kind"], "step": state["step"], "checkpoint": pin(path),
               "identity": state["identity"], "created_utc": now()}
    atomic_json(path.with_suffix(".json"), receipt)
    atomic_json(Path(folder) / "resume.json", receipt)
    if os.environ.get("WQQ_CAMPAIGN_CHECKPOINT_FILE"):
        atomic_json(Path(os.environ["WQQ_CAMPAIGN_CHECKPOINT_FILE"]),
                    {"checkpoint_path": relative(path), "sha256": receipt["checkpoint"]["sha256"]})
    return receipt


def latest_checkpoint(folder, identity):
    receipts = sorted((Path(folder) / "checkpoints").glob("step-*.json"), reverse=True)
    if not receipts:
        return None
    # A complete receipt is committed only after the corresponding atomic file.
    receipt = read_json(receipts[0])
    if receipt["identity"] != identity:
        raise ValueError("Existing checkpoint belongs to another scientific run")
    return torch.load(checked(receipt["checkpoint"]), map_location="cpu", weights_only=True)


def stop_requested():
    filename = os.environ.get("WQQ_CAMPAIGN_STOP_FILE")
    return bool(filename and Path(filename).exists())


class CheckpointedStop(RuntimeError):
    pass


def fit(which, work=DEFAULT_WORK, *, resume=False, full_permit=None, family="balanced"):
    work = Path(work)
    protocol = verify_protocol(work)
    expected = cohort(which)
    if family not in ("balanced", "control"):
        raise ValueError("Only balanced and matched control continuations are registered")
    if family == "control" and which != "full":
        raise ValueError("Control outer weights must reuse the verified v74 control")
    if which == "full":
        if full_permit is None:
            raise PermissionError("Full fit requires an independently selected OOF recipe")
        validate_nomination(full_permit, family, work)
    folder = (work / "control" if family == "control" else work) / f"models/{which}"
    if (folder / "freeze.json").exists():
        frozen = read_json(folder / "freeze.json")
        if frozen["protocol"] != pin(work / "protocol.json"):
            raise ValueError("Completed fit has a different protocol")
        verify_ancestry(frozen, expected)
        return frozen
    clean_seal = read_json(work / "clean-seal.json") if family == "balanced" else {"rows": {}}
    if family == "balanced":
        checked(clean_seal["protocol"])
    original = read_json(NATIVE_SEAL)["rows"]
    warm = load_bundle("v70", which, work)
    dataset = {"original": [], "clean": []}; metadata = []; targets = {}; references = []
    for group, rows in (("original", original), ("clean", clean_seal["rows"])):
        for row in rows.values():
            if row["match"] not in expected:
                continue
            if row["primary_matches"] or not row["coarse_raw_feature_exact"]:
                raise ValueError("Training feature provenance failed")
            key = (row["match"], row["half"])
            if key not in targets:
                truth_path = DATA / f"{key[0]}-{key[1]}.truth.npy"
                targets[key] = native.targets(np.load(truth_path, allow_pickle=False))
                references.append(pin(truth_path))
            with np.load(checked(row["cache"]), allow_pickle=False) as z:
                x = z["x"].astype(np.float16)
            with np.load(checked(row["coarse"]), allow_pickle=False) as z:
                coarse = context(warm.context_model, z[row.get("coarse_key", "x")], len(targets[key]["y"]))
            dataset[group].append({"x": x, "coarse": coarse, **targets[key]})
            metadata.append(row)
    clean_count = 2 * len(expected) if family == "balanced" else 0
    if len(dataset["original"]) != 10 * len(expected) or len(dataset["clean"]) != clean_count:
        raise ValueError("Incomplete physical-match observation groups")
    count = STEPS_PER_MATCH * len(expected)
    identity = {"held": which, "family": family, "fit_matches": expected, "steps": count,
                "protocol": pin(work / "protocol.json"),
                "clean_seal": pin(work / "clean-seal.json") if family == "balanced" else None,
                "original_seal": pin(NATIVE_SEAL), "warmstart": warm.metadata,
                "source_hashes": protocol["sources"], "reference_hashes": references,
                "data_hashes": [{"cache": r["cache"], "coarse": r["coarse"]} for r in metadata]}
    torch.manual_seed(SEED); random.seed(SEED); rng = np.random.default_rng(SEED)
    model = native.JointRecovery().train()
    model.load_state_dict(warm.model.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    history = []; loss_sum = 0.; loss_count = 0; prior_elapsed = 0.; start = 0
    checkpoint = latest_checkpoint(folder, identity)
    if checkpoint:
        if not resume:
            raise FileExistsError("Checkpoints exist; use --resume to continue their exact state")
        state = restore_state(checkpoint, model, ema, optimizer, rng, identity)
        start = state["step"]; history = state["history"]; loss_sum = state["loss_sum"]
        loss_count = state["loss_count"]; prior_elapsed = state["elapsed"]
    started = time.monotonic()
    for step in range(start, count):
        optimizer.param_groups[0]["lr"] = .0001 * (.1 + .9 * .5 * (1 + np.cos(np.pi * step / count)))
        batch = balanced_batch(dataset, rng, clean_probability=.5 if family == "balanced" else 0.)
        optimizer.zero_grad(set_to_none=True)
        error = native.loss(model, batch)
        if not torch.isfinite(error):
            raise FloatingPointError("Nonfinite balanced training objective")
        error.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite balanced training gradients")
        optimizer.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.mul_(.995).add_(current, alpha=.005)
        loss_sum += float(error.detach()); loss_count += 1
        if (step + 1) % SAVE_EVERY == 0 or step + 1 == count:
            elapsed = prior_elapsed + time.monotonic() - started
            history.append({"step": step + 1, "loss_window_mean": loss_sum / loss_count, "seconds": elapsed})
            loss_sum = 0.; loss_count = 0
            state = checkpoint_state(model, ema, optimizer, rng, step + 1, identity,
                                     history, loss_sum, loss_count, elapsed)
            save_checkpoint(folder, state)
            print(json.dumps({"stage": "fit", "family": family, "held": which, **history[-1]}), flush=True)
            if stop_requested():
                raise CheckpointedStop("Stopped after an atomic resumable checkpoint")
    model_path = folder / "model.pt"; raw_path = folder / "final-raw-model.pt"
    atomic_torch(model_path, ema.state_dict()); atomic_torch(raw_path, model.state_dict())
    freeze = {"kind": "wqq-campaign-native-fit-v1", "family": family, "held": which,
              "fit_matches": expected, "steps": count, "model": pin(model_path),
              "final_raw_model": pin(raw_path), "protocol": pin(work / "protocol.json"),
              "context_teacher": {"freeze": warm.metadata["context_freeze"], "model": warm.metadata["context_model"], "fit_matches": expected},
              "weight_initialization": {"freeze": warm.metadata["freeze"], "model": warm.metadata["model"], "fit_matches": expected},
              "history": history, "EMA_decay": .995, "rows": metadata,
              "reference_reads": [{"stage": "fit", "match": p["path"].split("/")[-1].split("-")[0], **p} for p in references],
              "training_identity": identity, "full_permit": pin(full_permit) if full_permit else None}
    atomic_json(folder / "freeze.json", freeze, immutable=True)
    return freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "cache-clean", "fit", "predict-logits"))
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--held", choices=tuple(FITS))
    parser.add_argument("--family", choices=("v70", "control", "v74control", "balanced"), default=None)
    parser.add_argument("--match", choices=TRAIN + ("J03WPY", "J03WQQ"))
    parser.add_argument("--half", choices=HALVES)
    parser.add_argument("--condition", choices=("clean", "corrupted"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--permit", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    threadpool_limits(1); torch.set_num_threads(1)
    guard = StageAccessGuard(args.stage, FITS.get(args.held, ()) if args.stage == "fit" else (), args.match)
    sys.addaudithook(guard)
    if args.stage == "predict-logits" and args.match == "J03WQQ":
        validate_inference_role(args.held, args.match, args.permit, args.family or "v70", args.work)
        guard.wqq_authorized = True
    try:
        if args.stage == "prepare":
            result = prepare(args.work)
        elif args.stage == "cache-clean":
            result = cache_clean(args.work)
        elif args.stage == "fit":
            if args.held is None:
                parser.error("fit requires --held")
            result = fit(args.held, args.work, resume=args.resume, full_permit=args.permit,
                         family=args.family or "balanced")
        else:
            if any(x is None for x in (args.held, args.match, args.half, args.condition, args.out)):
                parser.error("predict-logits requires --held --match --half --condition --out")
            result = predict_logits(args.family or "v70", args.held, args.match, args.half, args.condition,
                                    args.out, args.work, args.permit)
    except CheckpointedStop as exc:
        print(str(exc), flush=True)
        raise SystemExit(75) from exc
    print(json.dumps({"stage": args.stage, "complete": True, "kind": result.get("kind"),
                      "work": relative(args.work)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
