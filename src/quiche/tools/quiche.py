import pertpy as pt
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import anndata
import quiche as qu
from sketchKH import sketch
import logging
from muon import MuData
from joblib import Parallel, delayed
from numba import njit
from sklearn.base import BaseEstimator
from pandas.api.types import is_numeric_dtype
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
from typing import List, Optional, Union
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@njit
def compute_avg_abundance(neighbor_indices, abundance_matrix):
    n_neighbors = neighbor_indices.shape[0]
    n_features = abundance_matrix.shape[1]
    avg = np.zeros(n_features, dtype=np.float64)
    
    for i in range(n_features):
        total = 0.0
        for j in range(n_neighbors):
            total += abundance_matrix[neighbor_indices[j], i]
        avg[i] = total / n_neighbors if n_neighbors > 0 else 0.0
    return avg

@njit
def get_top_indices(row, nlargest, min_perc):
    top_indices = np.argsort(row)[-nlargest:][::-1] #descending order
    selected = []
    for idx in top_indices:
        if row[idx] > min_perc:
            selected.append(idx)
    selected_indices = np.full(nlargest, -1, dtype=np.int64)
    for i in range(len(selected)):
        selected_indices[i] = selected[i]
    return selected_indices

class QUICHE(BaseEstimator):
    """Performs spatial enrichment analysis using QUICHE
    """
    def __init__(
        self,
        adata: anndata.AnnData,
        labels_key: str = 'cell_cluster',
        spatial_key: str = 'spatial',
        fov_key: str = 'fov',
        patient_key: str = 'Patient_ID',
        segmentation_label_key: Optional[str] = 'label',
        verbose: Union[int, bool] = 1,
        **kwargs
    ):
        """
        Initializes the QUICHE object.

        Parameters
        ----------
        adata: anndata.AnnData
            annotated data object containing preprocessed single-cell data
        labels_key: str (default = 'cell_cluster')
            column in adata.obs containing cell phenotype information
        spatial_key: str (default = 'spatial')
            key in adata.obsm containing cell centroid coordinates
        fov_key: str (default = 'fov')
            column in adata.obs with field of view information
        patient_key: str (default = 'Patient_ID')
            column in adata.obs with patient-level identifiers
        segmentation_label_key: str (default = 'label')
            column in adata.obs with segmentation label information. If None, will create a unique ID for every cell
        verbose: int or bool (default = 1)
            verbosity level
        kwargs:
            additional keyword arguments

        Example usage
        ----------
        # initalize class with your single cell anndata object, while specifying parameters for enrichment testing 
        # (1) sample level information (e.g., fov_key)
        # (2) patient level information for statistical testing (e.g., Patient_ID)
        # (3) annotated spatial objects in each sample (e.g., labels_key)
        # (4) unique object IDs in each sample (e.g., segmentation_label_key)
        # (5) spatial coordinates of each object in a sample (e.g., spatial)

        quiche_op = qu.tl.QUICHE(adata = adata, labels_key = 'cell_cluster', spatial_key = 'spatial',
                                fov_key = 'fov', patient_key = 'Patient_ID', segmentation_label_key = 'label')
        """
        self.adata = adata
        self.labels_key = labels_key
        self.spatial_key = spatial_key
        self.fov_key = fov_key
        self.patient_key = patient_key
        self.segmentation_label_key = segmentation_label_key
        self.verbose = verbose
        self.kwargs = kwargs

        if isinstance(verbose, bool):
            self.verbose = int(verbose)
        elif isinstance(verbose, int):
            self.verbose = verbose
        else:
            raise ValueError("Verbose must be an integer or boolean.")

        logger.setLevel(logging.DEBUG if self.verbose > 1 else logging.INFO)
        self._check_params()

        self.adata_niche = None
        self.adata_niche_subsample = None
        self.mdata = None
        self.cells_nonn = None
        self.adata_func = None

    def _check_params(self):
        """
        Performs initial data checks and conversions.
        """
        if not isinstance(self.adata, anndata.AnnData):
            raise ValueError("Input data must be anndata.AnnData.")
        
        if self.segmentation_label_key is None:
            self.segmentation_label_key = 'label'
            self.adata.obs['label'] = [str(i) for i in range(0, len(self.adata.obs_names))]

        for key in [self.labels_key, self.fov_key, self.patient_key, self.segmentation_label_key]:
            if key not in self.adata.obs.columns:
                raise KeyError(f"'{key}' key is not in adata.obs.")

        if self.spatial_key not in self.adata.obsm:
            raise KeyError(f"'{self.spatial_key}' key is not in adata.obsm.")

        try:
            self.adata.obs[self.fov_key] = self.adata.obs[self.fov_key].astype('category')
            logger.info(f"Converted '{self.fov_key}' to categorical.")
        except Exception as e:
            logger.warning(f"Failed to convert '{self.fov_key}' to categorical: {e}")

        if is_numeric_dtype(self.adata.obs_names):
            logger.warning('Converting adata.obs_names to strings.')
            self.adata.obs_names = self.adata.obs_names.astype('str')

    def compute_spatial_niches(
        self,
        radius: int = 200,
        p: int = 2,
        n_neighbors: Optional[int] = None,
        khop: Optional[int] = None,
        min_cell_threshold: int = 3,
        coord_type: str = 'generic',
        delaunay: bool = False,
        n_jobs: int = -1
    ):
        """
        Computes spatial spatial niches according to spatial neighbors.

        Parameters
        ----------
        radius: int (default = 200)
            radius in pixels for bounding local niche detection
        p: int (default = 2)
            integer referring to distance metric. 1 = manhattan distance, 2 = Euclidean distance
        n_neighbors: int or None (default = None)
            number of nearest neighbors for spatial proximity detection
        khop: int or None (default = None)
            number of hops for local niche detection
        min_cell_threshold: int (default = 3)
            minimum number of neighbors for a niche cell type vector to be considered
        coord_type: str (default = 'generic')
            type of spatial coordinates
        delaunay: bool (default = False)
            whether to use Delaunay triangulation
        n_jobs: int (default = -1)
            number of tasks for parallelization

        Returns
        ----------
        None

        Example usage
        ----------
        # define a spatial niche around an index cell as the composition of 30 nearest neighbors bounded by fixed pixel radius of 200
        quiche_op.compute_spatial_niches(radius = 200, n_neighbors = 30, min_cell_threshold = 3)

        # define a spatial niche around an index cell as the composition of all cells within a fixed pixel radius of 200
        quiche_op.compute_spatial_niches(radius = 200, n_neighbors = None, min_cell_threshold = 3)
        """
        logger.info('Computing spatial niches...')
        if khop is not None:
            niche_df, _ = qu.tl.spatial_niches_khop(
                self.adata,
                radius = radius,
                p = p,
                n_neighbors = n_neighbors,
                khop = khop,
                min_cell_threshold = min_cell_threshold,
                labels_key = self.labels_key,
                spatial_key = self.spatial_key,
                fov_key = self.fov_key,
                n_jobs = n_jobs
            )
            
            self.adata_niche = anndata.AnnData(niche_df)
            self.adata_niche.obs = self.adata.obs.loc[niche_df.index, :]
        else:
            self.adata = qu.tl.compute_spatial_neighbors(
                self.adata,
                radius = radius,
                n_neighbors = n_neighbors,
                spatial_key = self.spatial_key,
                delaunay = delaunay,
                fov_key = self.fov_key,
                coord_type = coord_type
            )
            self.adata_niche, self.cells_nonn = qu.tl.compute_niche_composition(
                self.adata,
                labels_key = self.labels_key,
                min_cell_threshold = min_cell_threshold
            )

        non_zero_indices = np.where(pd.DataFrame(self.adata_niche.X).sum(1) != 0)[0]
        self.adata_niche = self.adata_niche[non_zero_indices, :].copy()
        self.adata = self.adata[non_zero_indices, :].copy()

    def subsample(
        self,
        sketch_size: Optional[int] = None,
        sketch_key: str = 'Patient_ID',
        gamma: int = 1,
        frequency_seed: int = 0,
        n_jobs: int = -1
    ):
        """
        Performs distribution-focused downsampling if sketch_size is specified: see https://dl.acm.org/doi/10.1145/3535508.3545539.

        Parameters
        ----------
        sketch_size: int or None (default = None)
            number of niches to select from each patient sample. If None, will use all. 
        sketch_key: str (default = 'Patient_ID')
            key in adata.obs containing patient-level information for downsampling
        gamma: float (default = 1)
            scale parameter for the normal distribution standard deviation in random Fourier frequency features
        frequency_seed: int (default = 0)
            random state parameter in downsampling
        n_jobs: int (default = -1)
            number of parallel jobs

        Returns
        ----------
        None

        Example usage
        ----------
        # perform distribution-focused downsampling by selecting a total of 2500 niches from all images from each patient
        quiche_op.subsample(sketch_size = 2500, sketch_key = 'Patient_ID', n_jobs = 8)
        """
        if self.adata_niche is None:
            raise RuntimeError("spatial niches have not been computed. call 'compute_spatial_niches' first.")

        if sketch_size is None:
            logger.info('Skipping distribution-focused downsampling.')
            self.adata_niche_subsample = self.adata_niche
        else:
            logger.info('Performing distribution-focused downsampling...')
            _, self.adata_niche_subsample = sketch(
                self.adata_niche,
                sample_set_key = sketch_key,
                gamma = gamma,
                num_subsamples = sketch_size,
                frequency_seed = frequency_seed,
                n_jobs = n_jobs
            )
            logger.info(f"Downsampled niches to {self.adata_niche_subsample.shape[0]} total subsamples.")

    def differential_enrichment(
        self,
        k_sim: int = 100,
        design: str = '~condition',
        model_contrasts: str = 'condition1-condition0',
        solver: str = 'edger',
        n_jobs: int = -1
    ):
        """
        Tests for differential spatial enrichment across conditions using graph neighborhoods following milo approach: https://www.nature.com/articles/s41587-021-01033-z.

        Parameters
        ----------
        k_sim: int (default = 100)
            number of nearest neighbors in niche similarity graph construction
        design: str (default = '~condition')
            design formula for differential analysis
        model_contrasts: str (default = 'condition1-condition0')
            contrasts for condition-specific testing
        solver: str (default = 'edger')
            solver to use for differential analysis
        n_jobs: int (default = -1)
            number of tasks for parallelization

        Returns
        ----------
        None

        Example usage
        ----------
        # test for differential spatial enrichment across relapse conditions. Here we are building a 100-nearest neighbor niche similarity graph across patients
        # design: formula written in limma/edgeR style defining the linear model; ~Relapse means the model will estimate a baseline (intercept) and the effect of Relapse
        # model_contrasts: string defining the specific hypothesis to test; Relapse1-Relapse0 compares relapse 1 group to relapse 0 group
        quiche_op.differential_enrichment(design = '~Relapse', model_contrasts = 'Relapse1-Relapse0', k_sim = 100)

        # test for differential spatial enrichment using a continuous covariate 
        # when the predictor is continuous, the coefficient is tested directly so model contrasts are set to None
        quiche_op.differential_enrichment(design = '~Survival', model_contrasts = None, k_sim = 100)
        """
        if self.adata_niche_subsample is None:
            self.adata_niche_subsample = self.adata_niche.copy()

        logger.info('Testing for differential spatial enrichment across conditions...')
        self.adata_niche_subsample = qu.tl.construct_niche_similarity_graph(
            self.adata_niche_subsample,
            k = k_sim,
            n_jobs = n_jobs
        )
        self.mdata = self.quicheDA(
            design = design,
            model_contrasts = model_contrasts,
            solver = solver
        )
        self.mdata = MuData({'expression': self.adata, 'spatial_nhood': self.mdata['spatial_nhood'], 'quiche': self.mdata['milo']})
        self.mdata['quiche'].var.loc[:, ['-log10(SpatialFDR)', '-log10(PValue)']] = -1*np.log10(self.mdata['quiche'].var.loc[:, ['SpatialFDR', 'PValue']]).values
        self.mdata['quiche'].var[self.mdata['spatial_nhood'].obs.columns] = self.mdata['spatial_nhood'].obs.values

    def quicheDA(
        self,
        design: str = '~condition',
        model_contrasts: str = 'condition1-condition0',
        solver: str = 'edger'
    ):
        """
        Perform condition-specific differential analysis using QUICHE.

        Parameters
        ----------
        design: str (default = '~condition')
            design formula for differential analysis
        model_contrasts: str (default = 'condition1-condition0')
            contrasts for condition-specific testing
        solver: str (default = 'edger')
            solver to use for differential analysis

        Returns
        ----------
        mdata: MuData
            annotated data object after differential analysis.
        """
        milo = pt.tl.Milo()
        mdata = milo.load(self.adata_niche_subsample, feature_key='spatial_nhood')
        mdata['spatial_nhood'].uns["nhood_neighbors_key"] = None
        mdata = qu.tl.build_milo_graph(mdata, feature_key='spatial_nhood')
        mdata = milo.count_nhoods(mdata, sample_col = self.patient_key, feature_key = 'spatial_nhood')
        milo.da_nhoods(mdata, design = design, model_contrasts = model_contrasts, feature_key = 'spatial_nhood', solver = solver)
        return mdata

    def annotate_niches(
        self,
        annotation_scheme: str = 'neighborhood',
        annotation_key: str = 'quiche_niche_neighborhood',
        nlargest: int = 3,
        min_perc: float = 0.1,
        n_jobs: int = -1
    ):
        """
        Label niches based on the specified labeling scheme.

        Parameters
        ----------
        annotation_scheme: str (default = 'neighborhood')
            scheme to use for labeling ('neighborhood' or 'fov')
        annotation_key: str (default = 'quiche_niche_neighborhood')
            column in mdata['quiche'].var for storing annotated niche neighborhoods
        nlargest: int (default = 3)
            number of top cell types to label niche neighborhoods
        min_perc: float (default = 0.1)
            minimum proportion for cell type to be considering in labeling
        n_jobs: int (default = -1)
            number of tasks for parallelization

        Returns
        ----------
        None

        Example usage
        ----------
        # annotate niche neighborhoods by the top 3 most abundant cell types using the niche similarity graph
        quiche_op.annotate_niches(nlargest = 3, annotation_scheme = 'neighborhood', annotation_key = 'quiche_niche_neighborhood')

        # annotate niche neighborhoods by the top 5 most abundant cell types using the fov
        quiche_op.annotate_niches(nlargest = 5, annotation_scheme = 'fov', annotation_key = 'quiche_niche_fov')
        """
        if self.mdata is None:
            raise RuntimeError("Differential enrichment testing has not been performed. call 'differential_enrichment' first.")

        logger.info(f"Annotating niches using scheme: {annotation_scheme}...")
        if annotation_scheme == 'neighborhood':
            annotations = self.compute_niche_abundance_neighborhood(
                nlargest = nlargest,
                min_perc = min_perc,
                n_jobs = n_jobs
            )
        elif annotation_scheme == 'fov':
            annotations = self.compute_niche_abundance_fov(
                nlargest = nlargest,
                min_perc = min_perc,
                n_jobs = n_jobs
            )
        else:
            raise ValueError("invalid annotation_scheme. choose 'neighborhood' or 'fov'.")

        self.mdata['quiche'].var[annotation_key] = annotations
        self.mdata['spatial_nhood'].obs[annotation_key] = annotations

        try:
            self.mdata['quiche'].var[annotation_key].loc[np.isin(self.mdata['quiche'].var['index_cell'], self.cells_nonn)] = 'unidentified'
        except:
            pass

    def compute_niche_abundance_neighborhood(
        self,
        nlargest: int = 3,
        min_perc: float = 0.1,
        n_jobs: int = -1
    ):
        """
        Label niches using neighborhood information.

        Parameters
        ----------
        nlargest: int (default = 3)
            number of top cell types to label niche neighborhoods
        min_perc: float (default = 0.1)
            minimum proportion for cell type to be considering in labeling
        n_jobs: int (default = -1)
            number of tasks for parallelization

        Returns
        ----------
        annotations : list
            list of annotation strings for each niche.
        """
        knn_mat = self.mdata['spatial_nhood'].obsp['connectivities'].tocsr()
        df_prop = self.mdata['spatial_nhood'].to_df()
        abundance_matrix = df_prop.values
        label_names = df_prop.columns.tolist()
        n_niches, _ = abundance_matrix.shape

        def process_niche(niche_idx):
            start_ptr = knn_mat.indptr[niche_idx]
            end_ptr = knn_mat.indptr[niche_idx + 1]
            neighbor_indices = knn_mat.indices[start_ptr:end_ptr]
            
            if neighbor_indices.size == 0:
                return ''  #no neighbors
            
            avg_abundances = compute_avg_abundance(neighbor_indices, abundance_matrix)
            top_indices = np.argsort(avg_abundances)[-nlargest:]
            selected_labels = [label_names[i] for i, v in zip(top_indices, avg_abundances[top_indices]) if v > min_perc]
            if not selected_labels:
                return ''
            sorted_labels = '__'.join(sorted(selected_labels))
            return sorted_labels

        annotations = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_niche)(i) for i in tqdm(range(n_niches), desc="Labeling Niches"))
        return annotations

    def compute_niche_abundance_fov(
        self,
        nlargest = 3,
        min_perc = 0.1,
        n_jobs = -1
    ):
        """
        Label niches based on field of view (FOV) information.

        Parameters
        ----------
        nlargest: int (default = 3)
            number of top cell types to label niche neighborhoods
        min_perc: float (default = 0.1)
            minimum proportion for cell type to be considering in labeling
        n_jobs: int (default = -1)
            number of tasks for parallelization

        Returns
        ----------
        annotations : list
            List of annotation strings for each niche.
        """
        df = self.mdata['spatial_nhood'].to_df()
        labels_names = df.columns.tolist()
        data = df.values

        def process_niche(row):
            selected_indices = get_top_indices(row, nlargest, min_perc)
            selected_labels = [labels_names[idx] for idx in selected_indices if idx != -1]
            if selected_labels:
                sorted_labels = '__'.join(sorted(selected_labels))
                return sorted_labels
            else:
                return 'unidentified'

        annotations = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_niche)(row) for row in tqdm(data, desc="Labeling Niches"))
        return annotations
    
    def compute_functional_expression(
        self,
        niches: Optional[List[str]] = None,
        annotation_key: str = 'quiche_niche_neighborhood',
        min_cell_threshold: int = 3,
        foldchange_key: str = 'logFC',
        markers: Optional[List[str]] = None,
        n_jobs: int = -1
    ):
        """
        Computes functional expression of cell types within specified niches.

        Parameters
        ----------
        niches: list or None (default = None)
            list of niches of interest
        annotation_key: str (default = 'quiche_niche_neighborhood')
            string specifying the column in mdata with labeled niche neighborhoods
        min_cell_threshold: int (default = 3)
            minimum number of nearest neighbors in a niche to for expression analysis to be considered
        foldchange_key: str (default = 'logFC')
            column in mdata['quiche'].var with predicted log fold change values
        markers: list of str or None (default = None)
            list of functional markers to include in expression analysis. if None, will use all
        n_jobs: int (default = -1)
            number of tasks for parallelization

        Returns
        ----------
        None

        
        Example usage
        ----------
        # compute the functional expression of cell types within outcome-associated niches 
        functional_markers = ['PDL1', 'Ki67', 'GLUT1', 'CD45RO', 'CD69', 'PD1', 'CD57', 'TBET', 'TCF1', 'CD45RB', 'TIM3', 'IDO', 'LAG3', 'CD38', 'HLADR']
        quiche_op.compute_functional_expression(niches = niches,
                                                annotation_key = 'quiche_niche_neighborhood',
                                                min_cell_threshold = 3,
                                                foldchange_key = 'logFC',
                                                markers = functional_markers,
                                                n_jobs = 8)
        """
        if self.mdata is None:
            raise RuntimeError("Differential enrichment testing has not been performed. call 'differential_enrichment' first.")
        
        if markers is None:
            markers = self.mdata['expression'].var_names.tolist()

        missing_markers = set(markers) - set(self.mdata['expression'].var_names)
        if missing_markers:
            raise ValueError(f"the following markers are not present in the data: {missing_markers}")

        if niches is None:
            raise ValueError("niches must be provided as a list of niches of interest.")

        fov_obs_names = self.mdata['spatial_nhood'].obs_names
        expression_obs_names = self.mdata['expression'].obs_names
        idx = expression_obs_names.get_indexer(fov_obs_names)
        conn_mat = self.mdata['expression'].obsp['spatial_connectivities'][idx, :]

        quiche_var = self.mdata['quiche'].var
        sig_bool = np.isin(quiche_var[annotation_key].values, niches)
        conn_mat = conn_mat[sig_bool, :]
        niche_list = quiche_var[annotation_key].values[sig_bool]

        if not isinstance(conn_mat, csr_matrix):
            conn_mat = csr_matrix(conn_mat)

        nn_array = [conn_mat.indices[conn_mat.indptr[i]:conn_mat.indptr[i+1]] for i in range(conn_mat.shape[0])]

        labels = self.mdata['expression'].obs[self.labels_key].values
        unique_cell_types = np.unique(labels)
        cell_clusters_indices = {cell_type: set(np.where(labels == cell_type)[0]) for cell_type in unique_cell_types}

        segmentation_labels = self.mdata['expression'].obs[self.segmentation_label_key].values
        fov_values = self.mdata['expression'].obs[self.fov_key].values
        expression_data = self.mdata['expression'][:, markers].X
        
        if isinstance(expression_data, csr_matrix):
            expression_data = expression_data.toarray()

        def process_niche(niche_idx):
            niche = niche_list[niche_idx]
            nn = nn_array[niche_idx]
            cell_types = niche.split('__')
            func_records = []
            for cell_type in cell_types:
                cell_type_indices = cell_clusters_indices.get(cell_type, set())
                if not cell_type_indices:
                    continue
                idx_cell_type_nn = list(set(nn).intersection(cell_type_indices))
                if len(idx_cell_type_nn) >= min_cell_threshold:
                    exp = expression_data[idx_cell_type_nn, :]
                    exp_df = pd.DataFrame(exp, columns=markers)
                    exp_df[annotation_key] = niche
                    exp_df[self.labels_key] = cell_type
                    exp_df[self.segmentation_label_key] = segmentation_labels[idx_cell_type_nn]
                    exp_df[f'{annotation_key}_cell_type'] = f"{niche}:{cell_type}"
                    exp_df[self.fov_key] = fov_values[idx_cell_type_nn]
                    func_records.append(exp_df)
            return func_records

        total_niches = len(nn_array)

        with tqdm_joblib(tqdm(total=total_niches, desc="Computing Functional Expression")):
            func_results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(process_niche)(i) for i in range(total_niches))

        func_arr = [df for sublist in func_results for df in sublist]

        if func_arr:
            func_df = pd.concat(func_arr, ignore_index=True)
        else:
            func_df = pd.DataFrame(columns=markers + [annotation_key, self.labels_key, self.segmentation_label_key, f'{annotation_key}_cell_type', self.fov_key])

        adata_func = anndata.AnnData(func_df.drop(columns = [annotation_key, self.labels_key, self.segmentation_label_key, f'{annotation_key}_cell_type', self.fov_key]))
        adata_func.obs = func_df.loc[:, [annotation_key, self.labels_key, f'{annotation_key}_cell_type', self.segmentation_label_key, self.fov_key]]
        adata_func.obs = pd.merge(adata_func.obs, pd.DataFrame(self.mdata['quiche'].var.groupby([annotation_key])[foldchange_key].mean()), on = [annotation_key]) ##average logFC of the niche neighborhood
        self.adata_func = adata_func