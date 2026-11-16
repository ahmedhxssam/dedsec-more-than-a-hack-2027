# Pinned runtime dependencies

Exact replay was verified with CPython 3.11.12 on macOS arm64, CPU only. Install the exact versions in `requirements.txt`.

| Package | Pinned version | Installed package license metadata |
| --- | --- | --- |
| filelock | 4.2.4 | MIT |
| fsspec | 2026.9.0 | BSD-3-Clause |
| jinja2 | 3.1.6 | See installed distribution license files |
| joblib | 1.5.2 | BSD 3-Clause |
| markupsafe | 3.0.4 | BSD-3-Clause |
| mpmath | 1.3.0 | BSD |
| networkx | 3.6.1 | BSD-3-Clause |
| numpy | 2.2.6 | Copyright (c) 2005-2024, NumPy Developers. |
| pandas | 2.3.3 | BSD 3-Clause License |
| python-dateutil | 3.0.0 | Dual License |
| pytz | 2026.5 | MIT |
| scikit-learn | 1.7.2 | BSD-3-Clause |
| scipy | 1.15.3 | Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers. |
| six | 1.17.0 | MIT |
| sympy | 1.14.0 | BSD |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| torch | 2.9.0 | BSD-3-Clause |
| typing-extensions | 4.16.0 | PSF-2.0 |
| tzdata | 2026.5 | Apache-2.0 |

The license column is a package-metadata excerpt, not a replacement for upstream notices. Installed wheels contain their own license files and third-party dependency notices. No dependency wheel is redistributed in this repository.

The frozen runtime additionally requires a local C compiler on 64-bit Linux. `reproduce.py` rebuilds the bundled decoder sources with `cc -O3 -ffp-contract=off -shared -fPIC … -lm`; macOS arm64 uses the supplied hash-pinned binaries. Linux byte parity remains unverified. Windows is not an admitted exact-replay platform.
