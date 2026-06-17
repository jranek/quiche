# Changelog

## Unreleased

### Added
- Thank you @tuhulab for providing the test suite.
- Modern test suite under `tests/` for import smoke, preprocessing, graph, metrics, and QUICHE smoke workflows.
- GitHub Actions CI matrix.

### Changed
- Support Python versions `3.9, 3.10`.
- Updated conda environment accordingly. 

### Fixed
- Followed PR by @tuhulab, thank you for finding these errors. Will reserve PR for modernization workflows.
- Fixed `compute_spatial_niches` khop parameter wiring (`k` vs `n_neighbors`).
- Fixed Delaunay error in `compute_spatial_niches` by converting `.obms['spatial']` to numpy array as mentioned in #12.
- Fixed obs alignment bug when filtering niche results.
- Fixed chained assignment in niche annotation path.
- Fixed undefined variable use in plotting default output path.