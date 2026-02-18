# Changelog

## Unreleased

### Added
- Modern test suite under `tests/` for import smoke, preprocessing, graph, metrics, and QUICHE smoke workflows.
- GitHub Actions CI matrix for Python 3.10/3.11/3.12.
- Migration report: `docs/migration-python-modernization.md`.

### Changed
- Packaging metadata modernized to PEP 621 in `pyproject.toml`.
- Supported Python versions updated to `>=3.10,<3.13`.
- Dependency model updated with extras: `plot`, `full`, `dev`.
- Conda environment (`venv_quiche.yml`) updated for modern Python/scientific stack compatibility while keeping R support.

### Fixed
- Replaced bare `except:` handlers with explicit exceptions.
- Fixed `compute_spatial_niches` khop parameter wiring (`k` vs `n_neighbors`).
- Fixed obs alignment bug when filtering niche results.
- Fixed chained assignment in niche annotation path.
- Fixed undefined variable use in plotting default output path.
- Improved optional dependency error messages for heavy workflows.
