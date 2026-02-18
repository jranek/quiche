import numpy as np
import scipy.sparse as sp
import pytest

from quiche.tools import graph


def test_construct_affinity_shape_and_symmetry():
    x = np.random.default_rng(2).normal(size=(10, 4))
    w = graph.construct_affinity(x, k=3, radius=1)
    assert w.shape == (10, 10)
    np.testing.assert_allclose((w - w.T).toarray(), np.zeros((10, 10)), atol=1e-8)


def test_get_igraph_requires_optional_dependency(monkeypatch):
    monkeypatch.setattr(graph, "ig", None)
    with pytest.raises(ImportError, match="python-igraph"):
        graph.get_igraph(sp.csr_matrix(np.eye(3)), directed=False)


def test_compute_spatial_neighbors_requires_squidpy(monkeypatch, synthetic_adata):
    monkeypatch.setattr(graph, "sq", None)
    with pytest.raises(ImportError, match="squidpy"):
        graph.compute_spatial_neighbors(synthetic_adata)


def test_spatial_niches_khop_runs(synthetic_adata):
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
