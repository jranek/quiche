# Changelog

## Unreleased

### Added
- Thank you @tuhulab for providing these tests.
- Modern test suite under `tests/` for import smoke, preprocessing, graph, metrics, and QUICHE smoke workflows.
- GitHub Actions CI matrix for Python 3.9/3.10/.

### Fixed
- Followed PR by @tuhulab, thank you for finding these errors. Will reserve PR for modernization workflows.
- Fixed `compute_spatial_niches` khop parameter wiring (`k` vs `n_neighbors`).
- Fixed obs alignment bug when filtering niche results.
- Fixed chained assignment in niche annotation path.
- Fixed undefined variable use in plotting default output path.