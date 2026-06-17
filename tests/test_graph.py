import numpy as np
import scipy.sparse as sp
import pytest

from quiche.tools import graph

def test_construct_affinity_shape_and_symmetry():
    x = np.random.default_rng(2).normal(size=(10, 4))
    w = graph.construct_affinity(x, k=3, radius=1)
    assert w.shape == (10, 10)
    np.testing.assert_allclose((w - w.T).toarray(), np.zeros((10, 10)), atol=1e-8)

def test_compute_spatial_neighbors(synthetic_adata):
    synthetic_adata = graph.compute_spatial_neighbors(synthetic_adata)
    assert 'spatial_connectivities' in synthetic_adata.obsp
    assert 'spatial_distances' in synthetic_adata.obsp

def test_spatial_niches_khop(synthetic_adata):
    niche_df, nn_dict = graph.spatial_niches_khop(
        synthetic_adata,
        radius=1000,
        k=2,
        khop=2,
        labels_key="cell_cluster",
        spatial_key="spatial",
        fov_key="fov",
        min_cell_threshold=1,
    )
    assert not niche_df.empty
    assert set(nn_dict.keys()) == {"fov1", "fov2"}
