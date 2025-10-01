import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Union, Dict
import anndata
import logging
import requests 
from tqdm import tqdm 
from pathlib import Path

def make_directory(directory: str = None):
    """Creates a directory at the specified path if one doesn't exist.

    Parameters
    ----------
    directory : str
        A string specifying the directory path

    Returns
    ----------
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def standardize(x: np.ndarray):
    """Standardizes data by removing the mean and scaling to unit variance.

    Parameters
    ----------
    x: pd.DataFrame (default = None)
        data matrix (dimensions = cells x features)

    Returns
    ----------
    X: pd.DataFrame
        standardized data matrix (dimensions = cells x features)
    """
    scaler = StandardScaler(with_mean = True, with_std = True)
    X = scaler.fit_transform(x)
    
    return X

def compute_percentile(df: pd.DataFrame,
                       p: Union[int, float]  = 70):
    scores = df.values
    scores = scores[~np.isnan(scores)]
    perc = np.percentile(scores, p)
    return perc

def create_single_positive_table(marker_vals: pd.DataFrame,
                                 threshold_list: Dict):
    """ Determine whether a cell is positive for a marker based on the provided threshold.

    Parameters
    ----------
        marker_vals (pd.DataFrame): dataframe containing the marker intensity values
        threshold_list (list): list of functional markers and their pre-determined thresholds

    Returns
    ----------
        pd.DataFrame:
            contains the marker intensities as well as the single positive marker data
    """
    # create binary functional marker table, append to anndata table
    for marker, threshold in threshold_list:
        marker_vals[marker] = (marker_vals[marker].values >= threshold).astype('int')

    return marker_vals

def filter_fovs(adata: anndata.AnnData,
                patient_key: str,
                threshold: int):
    """Filters samples according to the number of cells/niches specifed.

    Parameters
    ----------
    adata: (default = None)
        anndata object
    patient_key: str 
        string indicating filtering key
    threshold: int
        integer referring to the minimum number of niches per sample

    Returns
    ----------
    adata:
        filtered anndata object
    """
    n_niches = adata.obs[patient_key].value_counts(sort=False)
    adata = adata[~np.isin(adata.obs[patient_key], n_niches[n_niches < threshold].index)]   

    return adata

def download_data(id: str = 'nt_preprocessed',
                  base_url: str = 'https://zenodo.org/records/14290163/files',
                  dest_str: str = 'data',
                  overwrite: bool = False,
                  headers: dict = None):
    """
    Downloads data from the Zenodo repository
    
    Parameters
    ----------
        id: str (default = 'nt_preprocessed')
            string specifying the data identifier
        base_url: str (default = https://zenodo.org/records/14290163/files)
            string specifying the Zenodo url
        dest_str: str (default = 'data')
            string specifying the save_directory
        overwrite: bool (default = False)
            boolean speciifying whether to overwrite the file if already downloaded
        headers: dictionary (default = None)
            HTTP headers
    """
    try:
        download_url = os.path.join(base_url, id + '.h5ad?download=1')
        dest_file_path = Path(os.path.join(dest_str, id +'.h5ad'))
        dest_file_path = Path(dest_file_path)

        #check if the file already exists
        if dest_file_path.exists():
            if overwrite:
                logging.info(f"Overwriting existing file at {dest_file_path}")
                try:
                    dest_file_path.unlink()
                except Exception as e:
                    logging.error(f"Failed to delete existing file {dest_file_path}: {e}")
                    raise
            else:
                logging.info(f"File already exists at {dest_file_path}. Skipping download.")
                return 

        dest_file_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(download_url, stream=True, headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get('Content-Length', 0))

        with open(dest_file_path, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {dest_file_path.name}') as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
        logging.info(f'Downloaded: {download_url} to {dest_file_path}')

    except requests.exceptions.RequestException as e:
        logging.error(f'Error downloading {download_url}: {e}')
        raise
    except Exception as e:
        logging.error(f'Error handling the file {dest_file_path}: {e}')
        raise