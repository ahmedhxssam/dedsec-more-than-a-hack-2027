"""Generate the DedSec entry from the organizer's tactical ATD files, on CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify_environment(runtime):
    if platform.python_version() != runtime["python"]:
        raise RuntimeError(f"Use CPython {runtime['python']} for exact replay")
    for name, version in runtime["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"Install {name}=={version} from requirements.txt")


def build_kernels(chain, spec):
    """Rebuild original C sources on 64-bit Linux; macOS arm64 uses the pinned builds."""
    system, machine = platform.system(), platform.machine()
    if system == "Darwin" and machine == "arm64":
        return
    if system != "Linux" or machine not in {"x86_64", "aarch64"}:
        raise RuntimeError("Use macOS arm64 or 64-bit Linux")
    builds = ((chain.sprint_decoder, "decoder"), (chain.fast, "graph"))
    spec["compiled_kernels"]["kernels"] = []
    for module, stem in builds:
        original = module.BUILD / f"{stem}.c"
        if original.read_text() != module.SOURCE:
            raise ValueError(f"Unexpected kernel source: {original}")
        module.BUILD = ROOT / ".build" / stem
        module.BUILD.mkdir(parents=True, exist_ok=True)
        source, binary = (module.BUILD / f"{stem}.{ext}" for ext in ("c", "dylib"))
        source.write_bytes(original.read_bytes())
        subprocess.run(["cc", "-O3", "-ffp-contract=off", "-shared", "-fPIC", str(source),
                        "-o", str(binary), "-lm"], check=True)
        kernel = chain.kernel_files(module, stem)
        spec["compiled_kernels"]["kernels"].append(kernel)
        spec["runtime_pins"].extend((kernel["source"], kernel["binary"]))


def reproduce(tactical_dir, output):
    config = json.loads((ROOT / "inference.json").read_text())
    verify_environment(config["runtime"])
    for name, digest in config["files"].items():
        verify(ROOT / name, digest)
    tracking = tactical_dir / "BAR-ATH_tactical_atd.txt"
    metadata = tactical_dir / "BAR-ATH_tactical_atd.xml"
    verify(tracking, config["inputs"]["tracking"])
    verify(metadata, config["inputs"]["metadata"])

    from scripts import wqq_campaign_transition_replay_v1 as replay
    from second_zero.evaluate import validate_submission
    from second_zero.ingest import load_metadata, load_tracking
    from vendor.mtah import games

    np, chain = replay.np, replay.chain
    if config["policy"] != replay.POLICY:
        raise ValueError("Unexpected decoder policy")
    game = games.GAMES[games.GAME]
    blocks = [(h, a, b + 1) for h, a, b in games.half_video_blocks(games.GAME)]
    count, fps = game["n_video_frames"], game["fps"]
    meta = load_metadata(metadata)
    if (meta["n_frames"], meta["fps"]) != (count, fps):
        raise ValueError("Tracking metadata does not match the organizer's timeline")
    track = load_tracking(tracking, meta, count)
    pitch = np.array(game["pitch"], dtype=np.float64)
    team = np.array([0 if side == "H" else 1 if side == "A" else -1
                     for side in track.player_team], dtype=np.int64)
    keep = team >= 0
    predictions = np.full(count, 3, dtype=np.int8)
    build_kernels(chain, config["spec"])
    with chain.threadpool_limits(1):
        bundle = chain.load_bundle(config["spec"])
        for half, start, end in blocks:
            raw = {
                "ball": (track.ball_xy[start:end].astype(np.float64) * pitch).astype(np.float32),
                "players": (track.players_xy[start:end, keep].astype(np.float64) * pitch).astype(np.float32),
                "team": team[keep],
                "pitch": pitch,
            }
            arrays, _ = replay.predict_raw(raw, bundle)
            labels = arrays["transition"]
            if labels.shape != (end - start,) or not np.isin(labels, [0, 1, 2, 3]).all():
                raise ValueError(f"Invalid output for half {half}")
            predictions[start:end] = labels
            print(f"Half {half}: inferred {end - start:,} frames", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an existing submission, or publish a mismatched replay.
    with tempfile.TemporaryDirectory(prefix=".dedsec-", dir=output.parent) as tmp:
        staged = Path(tmp) / "predictions.csv"
        with staged.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("frame", "label"))
            writer.writerows((frame, "HADX"[int(label)]) for frame, label in enumerate(predictions))
        result = validate_submission(staged, count)
        verify(staged, config["output_sha256"])
        os.link(staged, output)
    print(f"Validated {result['rows']:,} rows and exact SHA-256 {config['output_sha256']}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tactical-dir", required=True, type=Path,
                     help="Directory containing the two organizer tactical ATD files")
    cli.add_argument("--output", required=True, type=Path, help="New CSV path, such as DedSec.csv")
    args = cli.parse_args()
    if args.output.exists():
        cli.error("--output already exists; use a new path")
    reproduce(args.tactical_dir, args.output)


if __name__ == "__main__":
    main()
