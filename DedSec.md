# DedSec — tracking-based team possession

**More Than a Hack 2027, Initial Challenge · 16 November 2026**  
**Submission method:** Strong1 (`strong_scale1`), the complete original tracking ensemble with native joint possession/status inference and typed transition scale **1**.

## Approach

We predict one of home possession (`H`), away possession (`A`), dead ball (`D`) or outside-play/unknown (`X`) for every tactical-video frame. The output contains all **199,319** frames in their original order. We process the organizer-defined halves separately and write `X` outside those blocks, including halftime. We do not concatenate halves across a stoppage or shorten the video to the available tracking.

The method combines two complementary sources of evidence. The retained parent ensemble uses ball/player geometry, observed motion, temporal ownership, contextual estimates around gaps and learned short-range ball imputation. Observed and inferred ball evidence remain distinct. A native temporal model jointly estimates team ownership, live/dead state and typed ownership-change, stop and restart hazards. These models use surrounding observations offline; this is not a causal live system.

The fixed composition mixes parent and native H/A/D state evidence equally (`alpha=0.5`, parent confidence `0.98`). A three-state Viterbi decoder uses the native typed transition energies at **scale 1**, the original node-energy division by five, the original native hazard correction and the original restart-destination prior. Initialization and tie order are unchanged. The same policy is applied throughout both halves; there are no manually specified possession intervals, game-specific label overrides or post-hoc CSV corrections. The repository's `second_zero/strong_scale1_candidate.py` is the short authoritative composition.

Strong1 retains **all 20 original serialized weight files** pinned by `inference.json`: ownership, interpolation, memory, carrier, contextual/live-dead, boundary, segment-quality, gap-recovery, verifier and native joint components. Some files bundle multiple estimators. Its scale-1 composition adds **no fitted weights**. In particular, the later player-only source-NN, graph/transport, transfer-pretrained and tracking-error auxiliary candidates are **not** dependencies of this submission.

## Provided assets: use and exclusions

| Asset | Use in this entry |
|---|---|
| `BAR-ATH_tactical_atd.txt` | Automatically produced ball/player XY observations on the complete tactical timeline. Missing observations remain missing. |
| `BAR-ATH_tactical_atd.xml` | Channel/team mapping, frame count and sampling metadata. Normalized XY is converted to metres using the official pitch dimensions. |
| Organizer `games.py` metadata | The declared 25 Hz timeline, pitch and inclusive half bounds; converted consistently to half-open array slices. |
| Tactical/broadcast video and audio | Not inference inputs. No challenge-video hand annotation is used. |
| Broadcast tracking, Smart Tagging, homographies and other event assets | Omitted from the final runtime. Alignment and camera/tracking reliability add uncertainty; the retained route needs only the two tactical files. |
| Supplied speed values | Not model inputs. Motion features are computed from the permitted XY observations. |
| Human annotations/corrected tracks/events about this provided game, from any source | Excluded from training, calibration, model selection and inference. No BAR human ground truth is used. |

The public five-minute reference is restricted to pipeline checks such as frame alignment, file format and label meaning. It is not training data or a tuning/validation set for this entry. No private full-match challenge ground truth, hidden possession counts or derivative of that ground truth is used. We do not redistribute the competition tracking, video or reconstructive tracking exports in the public repository.

## Training data and provenance

All labelled examples used to fit or calibrate the **retained submission assets** come from one external release: Bassek, Rein, Weber and Memmert, [*An integrated dataset of spatiotemporal and event data in elite soccer*](https://doi.org/10.6084/m9.figshare.28196177.v1), released with DFL authorization under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These are different physical games from Barcelona–Athletic. The source and licence remain accessible at the linked release; the inference repository supplies the trained weights and does not require a private dataset service.

The labels are the release's **DFL/provider** `BallStatus` and `BallPossession` fields, not DedSec human annotations. DedSec created no new manual labels for the retained models. We do not claim independent human adjudication or a verified manual-versus-automatic production breakdown for those upstream provider fields. The public preparation code maps `BallStatus=0` to `D`; for live frames it maps possession code 1 to `H`, code 2 to `A`, and unsupported/unknown combinations to `X`. It aligns each entity by the provider's frame number. Label fields are stored separately and never become inference features.

| Role in the retained method | Physical matches | Aligned 25 Hz frame slots, both halves |
|---|---|---:|
| Fit learned weights, including cross-fit ancestors | J03WMX, J03WN1, J03WOH, J03WOY | **567,278** |
| Inherited calibration and configuration selection | J03WPY | **146,211** |
| Repeated external development/audit | J03WQQ | **142,345** |
| Repeated, weight-excluded readiness audit | J03WR9 | **146,810** |

Frame totals include unknown/ungraded slots; they are not claims that every slot has a usable H/A/D label. Individual specialists use subsets or cross-fit pairs within the four fitting games. No retained weight is fitted on WPY, WQQ or WR9. WPY calibrates inherited temperatures, mixtures and decoder choices; WQQ and WR9 have subsequently informed research/readiness decisions and are **not untouched test matches**. The retained functional ancestry and all weight identities are recorded in the package manifests.

Training creates observation corruption on the external fitting games to simulate missing and noisy tracking. These are additional views of the same games, not additional labelled matches. Later development stress tests also used **unlabelled availability/repetition patterns** from the provided automatic tracking to simulate missing detections, stale positions and anonymous slot changes; they transferred observation-quality patterns, not human possession labels, corrected target positions or hand-produced events. Newer experiments using other external sources or six-game fine-tuning were rejected and their weights are excluded from Strong1.

## Possession rules and decoding

Possession means a team's sustained control, rather than simply its nearest player at each frame. The model may retain the current team through a pass, loose-ball interval or incidental opponent touch; a new spell needs sufficient learned ownership/transition evidence. A challenge or deflection does not automatically transfer possession. Conversely, short real controls are not deleted by a fixed minimum-duration rule.

`D` is a learned stopped-play state, not an automatic consequence of a missing ball detection or no nearby player. Stop and restart transitions are modelled separately. A restart opens a new live segment even if the same team had the previous possession. Outside the official half blocks the output is `X`; inside them the decoder normally emits H/A/D. These rules are learned approximations from provider supervision, not direct referee or contact observations, and provider conventions may differ from the challenge's annotation convention.

## Validation and limitations

The final CSV was regenerated in a fresh standalone process from the two original tactical files and packaged weights. It matched the expected SHA-256 exactly and passed the organizer-format checks. That replay took **230.338 seconds** on the verified local CPU environment. It read no reference-label arrays, cached features, cached predictions or external inference weights.

On the reused external **J03WR9** audit, Strong1 obtained clean SpellF1 **0.8129577465**, and **0.6563953488 / 0.6322580645** under two fixed target-shaped tracking corruptions; their noisy mean is **0.6443267067**. Each result uses the official five temporal-IoU thresholds. These are adaptive development measurements on an external match, not hidden Barcelona–Athletic accuracy. Selection is game-dependent: other retained research variants did better on WPY/WQQ. The internal realistic-noise goal of **0.75 was not achieved**. An earlier 0.784112 result used a different clean/legacy-corruption benchmark and is not evidence for that realistic-noise goal or this match's hidden score.

Long ball gaps, identity reassignment, airborne play, short contested controls and ambiguous stoppages remain difficult. Available XY cannot directly establish deliberate control, a referee's decision or contact intent. Synthetic corruption covers only part of real camera/tracking failure. With more time we would seek independently labelled **other-game** control/stoppage data and improve short-spell and missing-ball evidence, rather than tune on the provided game's public labels.

## Reproduction

Use the public repository root and **CPython 3.11.12**. Install `requirements.txt` in a fresh environment; the checked numerical versions are NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3, scikit-learn 1.7.2, threadpoolctl 3.6.0 and PyTorch 2.9.0. CPU inference is sufficient. The byte-exact replay was verified on macOS Apple Silicon; Linux C-kernel build support is included, but Linux byte parity has not been established.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B reproduce_strong_scale1.py \
  --tactical-dir /path/to/organizer/tactical \
  --output /path/to/new-output/DedSec.csv
.venv/bin/python vendor/mtah/evaluation/validate_submission.py \
  /path/to/new-output/DedSec.csv
```

The tactical directory must contain the original named TXT/XML files. Use a new output path: existing CSVs and receipts are never overwritten. `strong1.json` binds the policy and expected output; `inference.json` pins the 152 original assets and environment; `runtime-closure.json` and the additional-file pins cover the packaged runtime. Input hashes, module locations, full half identities, output identity and format validation are checked and a separate receipt is written. This reproduces inference from raw data; retraining every historical specialist is not required to regenerate the submission.

**Expected CSV:** header `frame,label`, frames `0…199318`, 199,319 data rows.  
**SHA-256:** `d74b990829aa0ec69a141400129d92df0f4d17573749b15696fd6a026adf7a44`.
