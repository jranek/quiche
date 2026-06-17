import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata
import pytest

from quiche.tools import metrics

class DummyQuicheOp:
    def __init__(self, mdata):
        self.mdata = mdata

def _build_quiche_var_adata() -> anndata.AnnData:
    adata = anndata.AnnData(np.zeros((1, 6)))
    adata.var = pd.DataFrame(
        {
            "quiche_niche_neighborhood": ["A__B", "A__B", "B__C", "B__C", "A__B", "B__C"],
            "Patient_ID": ["p1", "p2", "p1", "p2", "p3", "p3"],
            "condition": ["0", "1", "0", "1", "0", "1"],
            "logFC": [1.0, 0.8, -1.1, -0.9, 1.2, -1.0],
            "SpatialFDR": [0.01, 0.02, 0.03, 0.04, 0.03, 0.01],
            "PValue": [0.02, 0.03, 0.01, 0.05, 0.04, 0.02],
        },
        index=[f"n{i}" for i in range(6)],
    )
    return adata

def test_compute_niche_composition(synthetic_adata):
    n = synthetic_adata.n_obs
    connectivities = sp.csr_matrix(np.eye(n))
    synthetic_adata.obsp["spatial_connectivities"] = connectivities
    adata_niche, cells_nonn = metrics.compute_niche_composition(
        synthetic_adata,
        connectivities_key="spatial_connectivities",
        labels_key="cell_cluster",
        min_cell_threshold=1,
    )
    assert adata_niche.n_obs == n
    assert isinstance(cells_nonn, list)

def test_compute_niche_metadata_and_filter_niches():
    quiche_adata = _build_quiche_var_adata()
    op = DummyQuicheOp({"quiche": quiche_adata})

    niche_df = metrics.compute_niche_metadata(
        op,
        annotation_key="quiche_niche_neighborhood",
        patient_key="Patient_ID",
        condition_key="condition",
        niche_threshold=0,
        metrics=["logFC", "SpatialFDR", "PValue"],
    )
    assert not niche_df.empty
    assert "proportion_patients_niche" in niche_df.columns

    scores_df = metrics.filter_niches(
        op,
        thresholds={"logFC": {"median": [-0.5, 0.5]}, "SpatialFDR": {"median": 0.05}},
        min_niche_count=1,
        annotation_key="quiche_niche_neighborhood",
    )
    assert isinstance(scores_df, pd.DataFrame)

def test_run_milo_requires_optional_dependencies(monkeypatch, synthetic_adata):
    monkeypatch.setattr(metrics, "pt", None)
    with pytest.raises(ImportError, match="pertpy"):
        metrics.run_milo(synthetic_adata)

def test_differential_cell_type_abundance_binary(synthetic_adata):
    condition_dict = {"p1": "0", "p2": "1"}
    norm_counts, results_df = metrics.differential_cell_type_abundance(
        synthetic_adata,
        condition_dict=condition_dict,
        patient_key="Patient_ID",
        labels_key="cell_cluster",
        condition_key="condition",
        condition_type="binary",
        id1="0",
        id2="1",
    )
    assert "FDR_p_value" in results_df.columns
    assert "Patient_ID" in norm_counts.columns