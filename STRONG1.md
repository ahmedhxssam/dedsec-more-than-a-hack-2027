# Strong1 inference contract

The selected entry is `reproduce_strong_scale1.py`; the expected CSV SHA-256 is `d74b990829aa0ec69a141400129d92df0f4d17573749b15696fd6a026adf7a44` for all 199,319 original video frames.

For each complete official half, the runner reconstructs metric raw observations from tactical TXT/XML, calls the packaged complete parent and original hierarchical-joint-v70 native4 model, and combines the joint H/A/D nodes with alpha 0.5 and parent confidence 0.98. Typed off-diagonal transition energies use scale 1.0, including the original log(2) restart destination prior. The native node /5 factor, native hazard correction, Viterbi initialization and tie order are retained.

There is no additional player-only source-NN weight, source module or blend. The original parent retains all of its packaged specialists. Runtime inputs are raw ball, player, team and pitch observations; labels and intermediate feature/prediction arrays are excluded.

The original 152 pinned scientific source/model assets remain byte-identical to the verified package. Submission-facing documentation and selection-status metadata are updated in this separate staging directory. The Strong1 prediction helper and numerical operations are unchanged. The base manifest `inference.json` and base helper `reproduce.py` remain frozen for component verification; the primary README documents the Strong1 route.

The runner preserves missing input coordinates, infers the two halves separately, fills out-of-play video intervals with X, validates the complete output and refuses to replace an existing output. Hidden accuracy is unknown. See `DedSec.md` for training-data attribution and evaluation limitations.
