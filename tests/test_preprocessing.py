from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quiche.preprocessing import utils


def test_standardize_center_and_scale():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = utils.standardize(x)
    np.testing.assert_allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(out.std(axis=0), [1.0, 1.0], atol=1e-7)


def test_filter_fovs_filters_small_groups(synthetic_adata):
    filtered = utils.filter_fovs(synthetic_adata, patient_key="Patient_ID", threshold=7)
    assert filtered.n_obs == 0


def test_make_directory_requires_path():
    with pytest.raises(ValueError, match="directory must be provided"):
        utils.make_directory(None)


def test_create_single_positive_table_accepts_dict():
    marker_vals = pd.DataFrame({"m1": [0.1, 0.7], "m2": [0.8, 0.2]})
    threshold_list = {"m1": 0.5, "m2": 0.5}
    out = utils.create_single_positive_table(marker_vals.copy(), threshold_list)
    assert out["m1"].tolist() == [0, 1]
    assert out["m2"].tolist() == [1, 0]


def test_download_data_writes_file(monkeypatch, tmp_path: Path):
    class DummyResponse:
        headers = {"Content-Length": "4"}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=1024):
            yield b"ab"
            yield b"cd"

    def fake_get(url, stream=True, headers=None):
        assert "example_id.h5ad?download=1" in url
        return DummyResponse()

    monkeypatch.setattr(utils.requests, "get", fake_get)

    utils.download_data(
        id="example_id",
        base_url="https://example.org/files",
        dest_str=str(tmp_path),
        overwrite=True,
    )

    out_file = tmp_path / "example_id.h5ad"
    assert out_file.exists()
    assert out_file.read_bytes() == b"abcd"
