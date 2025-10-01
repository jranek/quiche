import pandas as pd
import numpy as np
import quiche as qu
from sketchKH import sketch
import anndata
from muon import MuData
import pertpy as pt
import scanpy as sc
from scipy.sparse import csr_matrix
from statsmodels.stats.multitest import multipletests
from scipy.stats import ranksums, spearmanr
from typing import Union, Optional, Dict, List

def compute_niche_composition(adata: anndata.AnnData,
                                connectivities_key: str = 'spatial_connectivities',
                                labels_key: str = 'cell_cluster',
                                min_cell_threshold: int = 3):
    """Computes niches according to proportion of cell types within spatial proximity

    Parameters
    ----------
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data 
    connectivities_key: str (default = 'spatial_connectivities')
        string referring to connectivities matrix in adata.obsp
    labels_key: str (default = 'cell_cluster')
        string referring to the column in adata.obs that contains cell phenotype labels
    min_cells: int (default = 3)
        integer referring to the number of nearest neighbors for a niche cell type proportion vector to be considered

    Returns
    ----------
    adata_niche: anndata.AnnData
        annotated data object containing niche-level information (dimensions = cells x cell types)
    cells_nonn: list
        list of cells that don't pass min threshold cutoff
    """
    connectivities = adata.obsp[connectivities_key].tocsr()
    connectivities.data = np.ones_like(connectivities.data)
    unique_labels = adata.obs[labels_key].cat.categories
    labels = adata.obs[labels_key]
    labels_codes = labels.cat.codes.values
    n_cells = adata.n_obs
    n_labels = len(unique_labels)
    
    one_hot = csr_matrix((np.ones(n_cells), (np.arange(n_cells), labels_codes)),
                         shape=(n_cells, n_labels))
    
    neighborhood_counts_sparse = connectivities.dot(one_hot)
    neighborhood_counts = neighborhood_counts_sparse.toarray()
    
    neighborhood_counts_df = pd.DataFrame(neighborhood_counts, 
                                          index=adata.obs_names, 
                                          columns=unique_labels)
    
    neighborhood_counts_df.fillna(0, inplace=True)
    neighborhood_freq = neighborhood_counts_df.div(neighborhood_counts_df.sum(axis=1), axis=0)
    
    total_counts = neighborhood_counts_df.sum(axis=1)
    cells_nonn = total_counts[total_counts < min_cell_threshold].index.tolist()
    
    adata_niche = anndata.AnnData(neighborhood_freq)
    adata_niche.obs = adata.obs.loc[neighborhood_freq.index, :]
    
    return adata_niche, cells_nonn

def compute_niche_metadata(quiche_op,
                           niches: Optional[List[str]] = None,
                           annotation_key: str = 'quiche_niche_neighborhood',
                           patient_key: str = 'Patient_ID',
                           condition_key: str = 'condition',
                           niche_threshold: int = 3,
                           condition_type: str = 'binary',
                           metrics: Optional[List[str]] = ['logFC', 'SpatialFDR', 'PValue']):
    """
    Computes niche-level metadata.

    Parameters
    ----------
    quiche_op 
        quiche class after fitting the model
    niches: List[str], optional
        list containing niches of interest. if None, will use all
    annotation_key: str (default = 'quiche_niche_neighborhood')
        column name in mdata object that contains niche annotations
    patient_key: str (default = 'Patient_ID')
        column name in mdata with patient-level information
    condition_key: str (default = 'condition')
        column name in mdata object with condition-level information
    niche_threshold: int (default = 3)
        minimum number of niches per patient sample to be considered positive
    condition_type: str (default = 'binary')
        type of condition comparison. if 'continuous', then condition information will be binarized based on the median.
    metrics: list (default = ['logFC', 'SpatialFDR', 'PValue'])
        list of metric columns in mdata['quiche'].var to be summarized

    Returns
    -------
    niche_df: pd.DataFrame
        dataframe containing computed niche proportions and related statistics.

    Example usage
    ----------
    # filter niches by median logFC < -1 or logFC > 1 and median spatialFDR < 0.05
    niche_metadata = qu.tl.compute_niche_metadata(quiche_op,
                                                niches = niches,
                                                annotation_key = 'quiche_niche_neighborhood',
                                                patient_key = 'Patient_ID',
                                                condition_key = 'Relapse',
                                                condition_type  = 'binary',
                                                metrics = ['logFC', 'SpatialFDR', 'PValue'])
    """
    mdata = quiche_op.mdata

    for col in metrics:
        if col not in mdata['quiche'].var.columns:
            raise KeyError(f"'{col}' not found in mdata['quiche'].var.")
    
    if condition_type == 'continuous':
        ##binarize according to the median and reassign
        condition_series = mdata['quiche'].var[condition_key].astype(float)
        median_val = condition_series.median()
        binary_key = f"{condition_key}_binary"
        mdata['quiche'].var[binary_key] = (condition_series >= median_val).astype(int)
        condition_key = binary_key

    count_df = mdata['quiche'].var.groupby([annotation_key, patient_key, condition_key]).size()
    count_df = count_df[count_df > niche_threshold].reset_index(name='count')

    if niches is None: #if niches is None, use all 
        niches = count_df[annotation_key].unique().tolist()
    else:
        niches = list(set(niches).intersection(count_df[annotation_key].unique()))
    
    count_df = count_df[count_df[annotation_key].isin(niches)]
    avg_counts = count_df.groupby([annotation_key, condition_key])['count'].mean().reset_index()
    avg_counts.rename(columns={'count': 'mean_niche_abundance'}, inplace=True)
    
    patient_counts = count_df.groupby([annotation_key, condition_key])[patient_key].nunique().reset_index()
    patient_counts.rename(columns={patient_key: 'n_patients_niche'}, inplace=True)
    
    patient_ids = count_df.groupby([annotation_key, condition_key])[patient_key].unique().reset_index()
    patient_ids.rename(columns={patient_key: 'patient_ids'}, inplace=True)
    
    niche_df = pd.merge(patient_counts, patient_ids, on=[annotation_key, condition_key], how='left')
    
    statistics = {}
    for stat in metrics:
        median_stat = mdata['quiche'].var.groupby(annotation_key)[stat].median()
        mean_stat = mdata['quiche'].var.groupby(annotation_key)[stat].mean()
        statistics[f'med_{stat}'] = median_stat
        statistics[f'mean_{stat}'] = mean_stat
    stats_df = pd.DataFrame(statistics)
    stats_df.reset_index(inplace=True)
    
    niche_df = pd.merge(niche_df, stats_df, on=annotation_key, how='left')
    niche_df = pd.merge(niche_df, avg_counts, on=[annotation_key, condition_key], how='left')
    
    total_patients = mdata['quiche'].var.groupby(condition_key)[patient_key].nunique().reset_index()
    total_patients.rename(columns={patient_key: 'n_patients_condition'}, inplace=True)
    
    niche_df = pd.merge(niche_df, total_patients, on=condition_key, how='left')
    niche_df['proportion_patients_niche'] = niche_df['n_patients_niche'] / niche_df['n_patients_condition']
    
    return niche_df

def filter_niches(quiche_op,
                  thresholds: Dict[str, Dict[str, Union[float, List[float]]]] = {'logFC': {'median' : [-0.5, 0.5]}, 'SpatialFDR': {'median' : 0.05}},
                  min_niche_count: int = 5,
                  annotation_key: str = 'quiche_niche_neighborhood'):
    """Filters signficiant niche neighborhoods

    Parameters
    ----------
    quiche_op 
        quiche class after fitting the model
    thresholds: dictionary (default = {'logFC': {'median' : [-0.5, 0.5]}, 'SpatialFDR': {'median' : 0.05}})
        dictionary specifying metric, aggregation measure, and thresholds for filtering
    min_niche_count: int (default = 5)
        minimum number of niche neighborhoods per group to be considered
    annotation_key: str (default = 'quiche_niche_neighborhood')
        string referring to the column in mdata['quiche'].var containing niche neighborhood labels

    Returns
    ----------
    scores_df: pd.DataFrame
        dataframe containing filtered annotated niche groups with metrics

    Example usage
    ----------
    # filter niches by median logFC < -1 or logFC > 1 and median spatialFDR < 0.05
    niche_scores = qu.tl.filter_niches(quiche_op,
                                        thresholds = {'logFC': {'median' : [-1, 1]}, 'SpatialFDR': {'median' : 0.05}},
                                        min_niche_count = 5,
                                        annotation_key = 'quiche_niche_neighborhood')
    """
    mdata = quiche_op.mdata
    agg_dict = {metric: list(stat_dict.keys())[0] for metric, stat_dict in thresholds.items()}
    scores_df = mdata['quiche'].var.groupby(annotation_key).agg(agg_dict)

    for metric, stat_dict in thresholds.items():
        _, threshold = list(stat_dict.items())[0]
        if isinstance(threshold, list) and len(threshold) == 2:
            lower_bound, upper_bound = threshold
            scores_df = scores_df[(scores_df[metric] < lower_bound) | (scores_df[metric] >= upper_bound)]
        elif (metric == 'SpatialFDR') | (metric == 'PValue'):
            scores_df = scores_df[scores_df[metric] <= threshold]
        else:
            raise ValueError(f"Invalid threshold format for metric '{metric}'.")
        
    if min_niche_count is not None:
        niche_counts = mdata['quiche'].var[annotation_key].value_counts()
        valid_niches = niche_counts[niche_counts >= min_niche_count].index
        scores_df = scores_df[scores_df.index.isin(valid_niches)]
    
    return scores_df

def run_milo(adata: anndata.AnnData,
             n_neighbors: int = 100,
             design: str = '~condition',
             model_contrasts: str = 'condition1-condition0',
             patient_key: str = 'Patient_ID',
             cluster_key: str = 'cell_cluster',
             feature_key: str = 'rna',
             prop: float = 0.1):
    """Performs differential cell type abundance analysis using Milo: https://www.nature.com/articles/s41587-021-01033-z, https://pertpy.readthedocs.io/en/latest/usage/tools/pertpy.tools.Milo.html

    Parameters
    ----------
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data 
    n_neighbors: int (default = 10)
        number of nearest neighbors
    design: str (default = '~condition')
        string referring to design
    model_contrasts: str (default = 'conditionA-conditionB')
        string for condition-specific testing
    patient_key: str (default = 'Patient_ID') 
        string in adata.obs containing patient-level information
    cluster_key: str (default = 'cell_cluster')
        string in adata.obs containing cell cluster annotations
    feature_key: str (default = 'rna')
        str referring to expression information
    sig_threshold: float (default = 0.05)
        threshold for significance
    prop: float (default = 0.1)
        float for downsampling

    Returns
    ----------
    mdata: mudata object
        annotated data object containing cell type abundance analysis
    """
    milo = pt.tl.Milo()
    mdata = milo.load(adata)
    sc.tl.pca(mdata[feature_key])
    sc.pp.neighbors(mdata[feature_key], n_neighbors = n_neighbors, use_rep = 'X_pca')
    milo.make_nhoods(mdata[feature_key], prop = prop)
    mdata[feature_key].uns["nhood_neighbors_key"] = None
    mdata = qu.tl.build_milo_graph(mdata, feature_key=feature_key)
    mdata = milo.count_nhoods(mdata, sample_col = patient_key)
    milo.da_nhoods(mdata, design=design, model_contrasts = model_contrasts)
    milo.annotate_nhoods(mdata, anno_col = cluster_key, feature_key = feature_key)
    mdata = MuData({'expression': mdata[feature_key], 'milo': mdata['milo']})

    return mdata

def differential_cell_type_abundance(adata: anndata.AnnData,
                                    condition_dict: Dict,
                                    patient_key: str = 'Patient_ID',
                                    labels_key: str = 'cell_cluster',
                                    condition_key: str = 'condition',
                                    condition_type: str = 'binary',
                                    id1: Union[str, int] = '0',
                                    id2: Union[str, int]= '1'):
    """
    Tests for differential abundance of cell types under conditions.

    Parameters
    ----------
    adata: anndata.AnnData (default = None)
        annotated data object with single-cell data
    condition_dict: dict
        dictionary mapping patient IDs to their condition labels
    patient_key: str (default = 'Patient_ID')
        column in adata.obs containing patient-level information
    labels_key: str (default = 'cell_cluster')
        column in adata.obs specifying cell phenotype-level information
    condition_key: str (default = 'condition')
        column in adata.obs specifying conition-level information
    condition_type: str (default = 'binary')
        whether condition comparison is binary or continuous
    id1: str or int (default = '0')
        label for the first condition (if binary)
    id2: str or int (default = '1')
        label for the second condition (if binary)

    Returns
    -------
    norm_counts: pd.DataFrame
        normalized counts per patient and cell type
    results_df: pd.DataFrame
        dataframe of statistical results including p-values, log2 fold changes, etc
    """
    counts = adata.obs.groupby([patient_key, labels_key]).size().unstack().copy()
    norm_counts = counts.div(counts.sum(axis=1), axis=0).reset_index()
    norm_counts[condition_key] = pd.Series(norm_counts[patient_key]).map(condition_dict)
    cell_type_columns = counts.columns
    
    results = {labels_key: [], 'p_value': [], 'stat_value': [], 'Log2FC': []}
    if condition_type == 'binary':
        for cell_type in cell_type_columns:
            group1 = norm_counts[norm_counts[condition_key] == id1][cell_type]
            group2 = norm_counts[norm_counts[condition_key] == id2][cell_type]
            mean_group1 = group1.mean()
            mean_group2 = group2.mean()
            if (mean_group1 > 0) or (mean_group2 > 0):
                log2fc = np.log2((mean_group1 + 1e-10) / (mean_group2 + 1e-10))
            else:
                log2fc = np.nan
            try:
                stat_val, p_value = ranksums(group1, group2)
            except ValueError:
                stat_val, p_value = np.nan, np.nan

            results[labels_key].append(cell_type)
            results['p_value'].append(p_value)
            results['stat_value'].append(stat_val)  # store the z-statistic
            results['Log2FC'].append(log2fc)
    
    elif condition_type == 'continuous':
        for cell_type in cell_type_columns:
            x = norm_counts[cell_type]
            y = norm_counts[condition_key]

            valid_mask = ~x.isna() & ~y.isna()
            if valid_mask.sum() < 3:
                rho, p_value = np.nan, np.nan
            else:
                rho, p_value = spearmanr(x[valid_mask], y[valid_mask])

            log2fc = np.nan
            results[labels_key].append(cell_type)
            results['p_value'].append(p_value)
            results['stat_value'].append(rho)
            results['Log2FC'].append(log2fc)

    pvals = results['p_value']
    fdr_corrected = multipletests(pvals, method='fdr_bh')[1]
    results_df = pd.DataFrame({labels_key: results[labels_key],
                               'p_value': results['p_value'],
                               'stat_value': results['stat_value'],
                               'Log2FC': results['Log2FC'],
                               'FDR_p_value': fdr_corrected,
                               '-log10(Adj. p-value)': -np.log10(fdr_corrected)})
    
    return norm_counts, results_df