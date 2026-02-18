# Python Modernization and Compatibility Migration

## Overview
This migration modernizes QUICHE for Python 3.10-3.12, updates packaging metadata, introduces dependency extras, hardens optional dependency handling, and adds automated tests/CI.

## Goals
- Remove Python-version and packaging friction from old metadata/pins.
- Preserve existing public module paths and class/function names.
- Keep heavy optional stacks optional (scanpy/squidpy/pertpy/muon/sketchKH and R/edgeR workflows).
- Add reproducible validation through tests and CI.

## What changed

### 1) Packaging and dependency model
Files:
- `pyproject.toml`
- `venv_quiche.yml`

Changes:
- Updated `requires-python` to `>=3.10,<3.13`.
- Converted project metadata to modern PEP 621 style (`authors`, `[project.urls]`, explicit `dependencies`).
- Added dependency extras:
  - `plot`: plotting dependencies.
  - `full`: heavy spatial/omics dependencies.
  - `dev`: test/lint/type tooling.
- Added pytest config in `pyproject.toml`.
- Modernized conda environment pins for Python 3.10-3.12 compatibility.
- Kept R dependencies in the conda environment (`r-base`, `rpy2`, related R libs).

### 2) Compatibility and runtime safety refactors
Files:
- `src/quiche/__init__.py`
- `src/quiche/plotting/__init__.py`
- `src/quiche/preprocessing/utils.py`
- `src/quiche/tools/graph.py`
- `src/quiche/tools/metrics.py`
- `src/quiche/tools/quiche.py`
- `src/quiche/plotting/plot.py`

Changes:
- Replaced bare `except:` blocks with explicit handling.
- Removed legacy boolean comparison patterns (`== True`).
- Added optional dependency guards with actionable `ImportError` messages.
- Fixed discovered runtime issues:
  - `compute_spatial_niches(khop=...)` now calls `spatial_niches_khop` with `k=...` (correct parameter name).
  - `compute_spatial_niches` now filters `self.adata` using `obs_names` alignment, avoiding index desynchronization.
  - Fixed chained assignment in niche annotation updates.
  - Fixed undefined `metric` variable in `plot_niches` default save path.
  - Fixed `generate_colors` list-input handling.
  - Fixed incorrect `compute_niche_metadata` call in plotting helper.
- Added scanpy availability guard in differential-expression plotting.

### 3) Tests and CI
Files added:
- `tests/conftest.py`
- `tests/test_imports.py`
- `tests/test_preprocessing.py`
- `tests/test_graph.py`
- `tests/test_metrics.py`
- `tests/test_quiche_smoke.py`
- `.github/workflows/tests.yml`

Coverage of new tests:
- import smoke tests for package/submodules.
- preprocessing utilities and download workflow (network mocked).
- graph utilities and optional dependency guards.
- metrics computations and optional dependency guards.
- QUICHE class smoke checks on synthetic data.

CI:
- GitHub Actions matrix on Python `3.10`, `3.11`, `3.12`.
- Installs `.[dev]` and runs pytest with coverage output.

## Verification results
Local verification (Python 3.11):
- `pytest -q`: **19 passed**.
- `pytest --cov=quiche --cov-report=term-missing`: **19 passed**.

Notes:
- A pandas `FutureWarning` appears in one statistical helper due `groupby(...).size().unstack()` default behavior change in future pandas versions. This does not affect current correctness.

## Backward compatibility notes
- Public import paths are preserved (`quiche.pp`, `quiche.tl`, `quiche.pl`).
- Heavy operations now fail fast with clear guidance if optional deps are missing, instead of failing at package import time.
- `pip install .` remains lightweight and Python-focused.
- R/edgeR workflows remain supported via the conda environment path.

## Known limitations
- Full differential enrichment/plotting workflows may require `.[full]` dependencies and R setup depending on solver/tooling used.
- `pip install .` intentionally does not bundle R/Rpy2 runtime.
