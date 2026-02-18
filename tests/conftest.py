import numpy as np
import pandas as pd
import anndata
import pytest


@pytest.fixture
def synthetic_adata() -> anndata.AnnData:
    n_cells = 12
    x = np.random.default_rng(0).normal(size=(n_cells, 4)).astype(float)
    obs = pd.DataFrame(
        {
            "cell_cluster": pd.Categorical(
                ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
            ),
            "fov": pd.Categorical(["fov1"] * 6 + ["fov2"] * 6),
            "Patient_ID": ["p1"] * 6 + ["p2"] * 6,
            "label": [str(i + 1) for i in range(n_cells)],
            "condition": ["0"] * 6 + ["1"] * 6,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    adata = anndata.AnnData(x, obs=obs)
    adata.obsm["spatial"] = np.random.default_rng(1).uniform(0, 100, size=(n_cells, 2))
    return adata
