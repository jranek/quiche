import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Union, Dict
import anndata
def make_directory(directory: str = None):
    """Creates a directory at the specified path if one doesn't exist.

    Parameters
    ----------
    directory : str
        A string specifying the directory path

    Returns
    -------
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def standardize(x: np.ndarray):
    """Standardizes data by removing the mean and scaling to unit variance.

    Parameters
    x: pd.DataFrame (default = None)
        data matrix (dimensions = cells x features)
    ----------

    Returns
    X: pd.DataFrame
        standardized data matrix (dimensions = cells x features)
    ----------
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
    Args:
        marker_vals (pd.DataFrame): dataframe containing the marker intensity values
        threshold_list (list): list of functional markers and their pre-determined thresholds

    Returns:
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
    adata: (default = None)
        anndata object
    patient_key: str 
        string indicating filtering key
    threshold: int
        integer referring to the minimum number of niches per sample
    ----------

    Returns
    adata:
        filtered anndata object
    ----------
    """
    n_niches = adata.obs[patient_key].value_counts(sort=False)
    adata = adata[~np.isin(adata.obs[patient_key], n_niches[n_niches < threshold].index)]   

    return adata