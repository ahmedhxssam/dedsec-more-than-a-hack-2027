# DedSec — Strong1 possession submission

This repository reproduces DedSec’s selected **Strong1** entry from the organizer’s tactical tracking TXT/XML and the packaged frozen weights. Its fixed transition scale is **1.0**. The inference entry point is **`reproduce_strong_scale1.py`**.

## Reproduce the CSV

Use **CPython 3.11.12**. Exact byte reproduction is verified on **macOS arm64**, using CPU inference and the pinned packages below. Keep the organizer’s data outside this repository.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce_strong_scale1.py --tactical-dir /path/to/tactical --output ../DedSec.csv
python vendor/mtah/evaluation/validate_submission.py ../DedSec.csv --rows 199319
shasum -a 256 ../DedSec.csv
```

The tactical directory must contain `BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_atd.xml`. The output path and its `.receipt.json` sidecar must be new. The program verifies the environment, packaged source and weight hashes, tactical input hashes, complete frame timeline, output format and expected checksum before creating the final CSV.

Expected output: **199,319 data rows**, frames `0` through `199318`, columns `frame,label`, labels `H`, `A`, `D`, `X`.

```text
d74b990829aa0ec69a141400129d92df0f4d17573749b15696fd6a026adf7a44
```

A prior fresh standalone replay took 230.34 seconds on the development Mac; runtime varies by machine. Output-format validity and exact reproduction do not establish hidden-match accuracy. A score of 0.75 has not been demonstrated.

## Inference and manifests

Strong1 combines the preserved complete parent pipeline with the original hierarchical-joint-v70 model: alpha `0.5`, parent confidence `0.98`, joint status, and typed transition scale `1.0`. It uses each complete physical half independently. The additional player-only source-NN branch is excluded; all original parent specialists remain packaged.

- `strong1.json` identifies this selected policy, its expected CSV and file hashes.
- `inference.json` is the **preserved base-component manifest**. Its scale-0.5 policy and `95ac…` checksum describe the historical base route imported by the Strong1 entry; they are not the selected output. `reproduce.py` remains the base helper.
- `WEIGHTS.json` inventories all 20 serialized weight files. Some files contain multiple fitted estimators.
- `runtime-closure.json` and `SUBMISSION.json` identify the selected runtime and package contents.
- [DedSec.md](DedSec.md) describes the method, labeled-data provenance, experimental limitations and training scope.

No training is required to reproduce the entry. Inference needs no labels, event tags, video, broadcast tracking, source-dataset download, cached features or previous predictions. Organizer tracking and reference data are excluded from this repository.

## Other platforms

The preserved runner supports 64-bit Linux (`x86_64` or `aarch64`) by rebuilding the two bundled C decoder sources with a local C compiler:

```text
cc -O3 -ffp-contract=off -shared -fPIC SOURCE.c -o BINARY -lm
```

That build happens automatically during inference under `.build/`. The sources are `runs/two-hour-sprint/decoder/decoder.c` and `runs/metric-decoder-v11/kernel/graph.c`. Linux byte parity has not been verified; the output checksum check fails closed if a platform produces different bytes. The supplied macOS arm64 binaries remain hash-pinned. Other platforms require a separately verified port.

## Package checks and attribution

```bash
python -m unittest discover -s tests -p test_submission_metadata.py
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [DEPENDENCIES.md](DEPENDENCIES.md) for source and dependency notices. The organizer’s vendored timeline and metric/validator code retain their [MIT license](vendor/mtah/LICENSE). The external training dataset is attributed in `DedSec.md`; no dataset is redistributed here.
