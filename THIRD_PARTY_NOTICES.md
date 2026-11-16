# Third-party notices

## Organizer code

`vendor/mtah/games.py` and `vendor/mtah/evaluation/` are from Metrica Sports’ [More Than a Hack 2027 repository](https://github.com/metrica-sports/more-than-a-hack-2027). Their original MIT notice is preserved at `vendor/mtah/LICENSE`. These files supply the published frame timeline, output validator and possession metric.

## External training data

The trained components use the DFL-authorized release by Bassek, Rein, Weber and Memmert, *An integrated dataset of spatiotemporal and event data in elite soccer*, [Figshare version 1](https://doi.org/10.6084/m9.figshare.28196177.v1), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Provider BallStatus/BallPossession fields supply annotations; the detailed matches, roles and processing are documented in `DedSec.md`. Source data, labels and tracking records are not redistributed in this repository.

## Installed software

Pinned packages in `requirements.txt` are installed separately and retain their respective licenses. `DEPENDENCIES.md` records the package metadata and tested runtime. This notice does not replace upstream license terms.

## DedSec materials

The remaining original code and trained parameters are the DedSec submission materials. No additional third-party dataset or model has been added to the selected Strong1 runtime. The repository makes those materials available for challenge review and reproduction; separate upstream terms remain applicable to their own materials.
