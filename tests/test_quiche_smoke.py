import pytest

from quiche.tools.quiche import QUICHE
import quiche.tools.quiche as quiche_module

def test_quiche_init_requires_keys(synthetic_adata):
    bad = synthetic_adata.copy()
    bad.obs = bad.obs.drop(columns=["cell_cluster"])
    with pytest.raises(KeyError, match="cell_cluster"):
        QUICHE(bad)

def test_quiche_compute_spatial_niches_khop_smoke(synthetic_adata):
    op = QUICHE(synthetic_adata)
    op.compute_spatial_niches(radius=1000, n_neighbors=2, khop=2, min_cell_threshold=1)
    assert op.adata_niche is not None
    assert op.adata_niche.n_obs > 0

def test_quiche_subsample(synthetic_adata):
    op = QUICHE(synthetic_adata)
    op.compute_spatial_niches(radius=1000, n_neighbors=2, khop=2, min_cell_threshold=1)
    op.subsample(sketch_size=2)

# def test_quiche_da(synthetic_adata):
#     op = QUICHE(synthetic_adata)
#     op.adata_niche_subsample = synthetic_adata.copy()
#     op.quicheDA()