# Changelog

## Unreleased

### June 23, 2026

#### Added
- Added the ability to annotate all cells within the data.
- Thank you @tuhulab for providing the initial test suite located under `tests/`.
- GitHub Actions CI matrix.

#### Changed
- Support Python versions `3.9, 3.10`.
- Updated conda environment accordingly. 
- Modified `annotate_niches`, `plot_niches`, and `plot_niche_scores` to predict the annotations for cells not within the subsample.
- Modified `plot_niche_scores` to include `vcenter`. 
- Modified `compute_functional_expression` to accept predicted out of sample annotations. 
- Modified `compute_niche_network` and `plot_niche_network_donut` to enable sample-level scaling. 

#### Fixed
- Followed PR by @tuhulab, thank you for finding some these errors. Will reserve PR for modernization workflows.
- Fixed `compute_spatial_niches` khop parameter wiring (`k` vs `n_neighbors`).
- Fixed Delaunay error in `compute_spatial_niches` by converting `.obms['spatial']` to numpy array as mentioned in #12.
- Fixed obs alignment bug when filtering niche results.
- Fixed chained assignment in niche annotation path.
- Fixed undefined variable use in plotting default output path.
- Fixed `beeswarm` and `beeswarm_proportion` colors_dict order.