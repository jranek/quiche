import numpy as np
import pandas as pd
import igraph as ig
import scipy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import anndata
import squidpy as sq
import logging
import networkx as nx
from itertools import combinations
from typing import Union, Optional, Dict

def get_igraph(W: np.ndarray = None,
               directed: bool = None):
    """Converts adjacency matrix into igraph object

    Parameters
    ----------
    W: (default = None)
        adjacency matrix
    directed: bool (default = None)
        whether graph is directed or not

    Returns
    ----------
    g: ig.Graph
        graph of adjacency matrix
    """
    sources, targets = W.nonzero()
    weights = W[sources, targets]
    if type(weights) == np.matrix:
        weights = weights.A1 #flattens 
    g = ig.Graph(directed = directed)
    g.add_vertices(np.shape(W)[0])
    g.add_edges(list(zip(sources, targets)))
    g.es['weight'] = weights  

    return g

def heat_kernel(dist: np.ndarray = None,
                radius: int = 3):
    """Transforms distances into weights using heat kernel

    Parameters
    ----------
    dist: np.ndarray (default = None)
        distance matrix (dimensions = cells x k)
    radius: np.int (default = 3)
        defines the per-cell bandwidth parameter (distance to the radius nn)

    Returns
    ----------
    s: np.ndarray
        array containing between cell similarity (dimensions = cells x k)
    """         
    sigma = dist[:, [radius]]  # per cell bandwidth parameter (distance to the radius nn)
    s = np.exp(-1 * (dist**2)/ (2.*sigma**2)) # -||x_i - x_j||^2 / 2*sigma_i**2

    return s

def construct_affinity(X: np.ndarray,
                        k: int = 10,
                        radius: int = 3,
                        n_pcs: Optional[int] = None,
                        random_state: int = 0, 
                        n_jobs: int = -1):
    """Computes between cell affinity knn graph using heat kernel
    
    Parameters
    ----------
    X: np.ndarray (default = None)
        Data (dimensions = cells x features)
    k: int (default = None)
        Number of nearest neighbors
    radius: int (default = 3)
        Neighbor to compute per cell distance for heat kernel bandwidth parameter
    n_pcs: int (default = None)
        number of principal components to compute pairwise Euclidean distances for between-cell affinity graph construction. If None, uses adata.X
    n_jobs: int (default = -1)
        Number of tasks  

    Returns
    ----------
    W: np.ndarray
        sparse symmetric matrix containing between cell similarity (dimensions = cells x cells)
    """
    if n_pcs is not None:
        n_comp = min(n_pcs, X.shape[1])
        pca_op = PCA(n_components=n_comp, random_state = random_state)
        X_ = pca_op.fit_transform(X)
    else:
        X_ = X.copy()

    # find kNN
    knn_tree = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='euclidean', n_jobs=n_jobs).fit(X_)
    dist, nn = knn_tree.kneighbors()  # dist = cells x knn (no self interactions)

    # transform distances using heat kernel
    s = heat_kernel(dist, radius = radius) # -||x_i - x_j||^2 / 2*sigma_i**2
    rows = np.repeat(np.arange(X.shape[0]), k)
    cols = nn.reshape(-1)
    W = scipy.sparse.csr_matrix((s.reshape(-1), (rows, cols)), shape=(X.shape[0], X.shape[0]))

    # make symmetric
    bigger = W.transpose() > W
    W = W - W.multiply(bigger) + W.transpose().multiply(bigger)

    return W

def spatial_niches_khop(adata: anndata.AnnData,
                        radius: int = 200,
                        p: int = 2,
                        k: int = 10,
                        khop: int = 3,
                        labels_key: str = 'cell_cluster',
                        spatial_key: str = 'spatial',
                        fov_key: str = 'fov',
                        min_cell_threshold: int = 0,
                        n_jobs: int = -1):
    """Computes niches according to khop spatial neighborhood

    Parameters
    ----------
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data 
    radius: int (default = 200)
        integer referring to the radius in pixels for bounding local niche detection
    p: int (default = 2)
        integer referring to distance metric. 1 = manhattan distance, 2 = Euclidean distance
    k: int (default = 10)
        number of nearest neighbors
    khop: int (default = 3)
        number of k-hops
    labels_key: str (default = 'cell_cluster')
        string referring to the column in adata.obs that contains cell phenotype labels
    spatial_key: str (default = 'spatial') 
        string in adata.obsm containing cell centroid coordinates
    fov_key: str (default = 'fov')
        string in adata.obs containing sample-level annotations. Niches are defined locally in each ample
    min_cell_threshold: int (default = 0)
        integer referring to the number of nearest neighbors for a niche cell type proportion vector to be considered
    n_jobs: int (default = -1)
        number of tasks for parallelization

    Returns
    ----------
    niche_df: pd.DataFrame
        dataframe containing niches (dimensions = niche x cell type)
    nn_dict: dictionary
        dictionary containing nearest neighbor information
    """
    niche_df = []
    cells2remove = []
    nn_dict = {}
    for fov in np.unique(adata.obs[fov_key]):
        adata_fov = adata[np.where(adata.obs[fov_key] == fov)[0], :].copy()
        if adata_fov.shape[0] < k*khop:
            logging.info(f'{fov} is less than khop neighborhood size {k*khop}. Removing FOV.')
            cells2remove.append(list(adata_fov.obs_names))
        else:
            niche_df_fov = pd.DataFrame()
            knn_kdtree = NearestNeighbors(n_neighbors= khop*k, p = p, algorithm='kd_tree', metric='euclidean', n_jobs=n_jobs).fit(adata_fov.obsm[spatial_key])
            dist, nn = knn_kdtree.kneighbors()
            nn_dict[fov] = nn
            for i in range(0, nn.shape[0]):
                niche_nn = nn[i, :][dist[i, :] < radius]
                if len(niche_nn) < min_cell_threshold:
                    cells2remove.append(adata_fov.obs_names[i])
                else:
                    counts = adata_fov.obs[labels_key].iloc[niche_nn].value_counts().copy()
                    niche_df_fov_ = pd.DataFrame(counts / counts.sum()).transpose()
                    niche_df_fov = pd.concat([niche_df_fov, niche_df_fov_], axis = 0)
            niche_df_fov.index = adata_fov.obs_names[~np.isin(adata_fov.obs_names, cells2remove)]
            niche_df.append(niche_df_fov)
    try:
        cells2remove = np.concatenate(cells2remove)    
    except:
        pass
    niche_df = pd.concat(niche_df, axis = 0)
    niche_df = niche_df.fillna(0).copy()
    niche_df = niche_df.loc[adata.obs_names[~np.isin(adata.obs_names, list(set(cells2remove)))]]

    return niche_df, nn_dict

def construct_niche_similarity_graph(adata: anndata.AnnData,
                                     k: int = 100,
                                     n_jobs: int = -1):
    """Constructs niche similarity graph

    Parameters
    ----------
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data 
    k: int (default = 10)
        number of nearest neighbors in niche similarity graph construction
    n_jobs: int (default = -1)
        number of tasks for parallelization

    Returns
    ----------
    adata: anndata.AnnData
        annotated data object containing niche similarity graph
    """
    nn = NearestNeighbors(n_neighbors = k, algorithm = 'kd_tree', n_jobs = n_jobs).fit(adata.X)
    connectivities = nn.kneighbors_graph(mode = 'connectivity')
    distances = nn.kneighbors_graph(mode = 'distance')            
    adata.obsp['connectivities'] = connectivities
    adata.obsp['distances'] = distances
    adata.uns["nhood_neighbors_key"] = None
    adata.uns['neighbors'] = {'connectivities_key': 'connectivities',
                              'distances_key': 'distances',
                              'params': {'n_neighbors': k, 'method': 'kd_tree', 'distance': 'Euclidean'}}
    return adata

def build_milo_graph(mdata,
                     feature_key: str = 'spatial_nhood'):
    """Adds metadata from niche similarity graph"""
    #define idxs as all cells
    mdata[feature_key].obs['nhood_ixs_random'] = 1 
    mdata[feature_key].obs['nhood_ixs_refined'] = 1
    #binary knn graph
    knn_graph = mdata[feature_key].obsp['connectivities'].copy()
    knn_graph[knn_graph!=0] =1 #binarize
    mdata[feature_key].obsm['nhoods'] = knn_graph.copy()
    #distance matrix
    knn_dists = mdata[feature_key].obsp["distances"]
    knn_dists = knn_dists.max(1).toarray().ravel()
    mdata[feature_key].obs["nhood_kth_distance"] = knn_dists

    return mdata

def set_elements_to_zero_lil(sparse_array, mask):
    """Sparsify"""
    sparse_array = sparse_array.tolil()
    mask = mask.tolil()
    nz = mask.nonzero()
    sparse_array[nz] = 0

    return sparse_array.tocsr()

def bound_radius(adata: anndata.AnnData,
                distances_key: str = 'spatial_distances',
                connectivities_key: str = 'spatial_connectivities',
                radius: int = 200):
    """Bounds spatial similarity by a fixed pixel radius

    Parameters
    ----------
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data 
    distances_key: str (default = 'spatial_distances')
        string referring to distance matrix in adata.obsp
    connectivities_key: str (default = 'spatial_connectivities')
        string referring to connectivities matrix in adata.obsp
    radius: int (default = 200)
        integer referring to the number of pixels for bounding

    Returns
    ----------
    adata: anndata.AnnData
        annotated data object
    """
    dist_mat = adata.obsp[distances_key]
    connect_mat = adata.obsp[connectivities_key]
    bool_idx = dist_mat >= radius
    connect_mat = set_elements_to_zero_lil(connect_mat, bool_idx)
    adata.obsp[connectivities_key] = connect_mat

    return adata

def compute_spatial_neighbors(adata: anndata.AnnData,
                            radius: int = 200,
                            n_neighbors: int = 30,
                            spatial_key: str = 'spatial',
                            delaunay: bool = False,
                            fov_key: str = 'fov',
                            coord_type: str = 'generic'):
    """Computes spatial proximity graph according to knn, delaunay, radius

    Parameters
    ----------
    adata: anndata.AnnData (default = None)
        annotated data object containing preprocessed single-cell data 
    radius: int (default = 200)
        integer referring to the radius in pixels for bounding local niche detection
    n_neighbors: int (default = 30)
        number of nearest neighbors for local niche detection. If None, either delaunay or all cells in radius will be used
    delaunay: bool (default = False)
        boolean specifying whether delaunay triangulation should be used
    spatial_key: str (default = 'spatial') 
        string in adata.obsm containing cell centroid coordinates
    fov_key: str (default = 'fov)
        string in adata.obs containing sample-level annotations. Niches are defined locally in each ample
    coord_type: str (default = 'generic')
        string referring to spatial layout

    Returns
    ----------
    adata: anndata.AnnData
        annotated data object containing spatial proximity graph
    """
    adata.obs[fov_key] = pd.Categorical(adata.obs[fov_key])
    if n_neighbors is not None:
        sq.gr.spatial_neighbors(adata, spatial_key = spatial_key, library_key = fov_key, n_neighs = n_neighbors, coord_type = coord_type)
        adata = bound_radius(adata, distances_key = 'spatial_distances', connectivities_key = 'spatial_connectivities', radius = radius)
    elif delaunay == True:
        sq.gr.spatial_neighbors(adata, spatial_key = spatial_key, library_key = fov_key, delaunay = delaunay, coord_type = coord_type)
        adata = bound_radius(adata, distances_key = 'spatial_distances', connectivities_key = 'spatial_connectivities', radius = radius)
    else:
        sq.gr.spatial_neighbors(adata, spatial_key = spatial_key, library_key = fov_key, radius = radius, coord_type = coord_type)
        
    return adata

def compute_niche_network(niche_df: pd.DataFrame,
                          colors_dict: Dict,
                          lineage_dict: Dict,
                          annotation_key: str = 'quiche_niche_neighborhood'):
    """Computes niche network. Cell types are connected to one another according to the number of unique patients with corresponding interaction

    Parameters
    ----------
    niche_df: pd.DataFrame (default = None)
        dataframe containing niche-level metadata 
    colors_dict: dictionary (default = None)
        dictionary containing cell type colors
    lineage_dict: dictionary (default = None)
        dictionary containing lineage colors
    annotation_key: str (default = 'quiche_niche_neighborhood')
        string referring to the column in niche_df with niche annotations

    Returns
    ----------
    G: nx.Graph
        networkx graph containing cell type interactions 
    """
    node_dict = {}
    for index, row in niche_df.iterrows():
        nodes = row[annotation_key].split('__')
        for node in nodes:
            if node in node_dict:
                node_dict[node]['mean_logFC'].append(row['mean_logFC'])
            else:
                node_dict[node] = {'mean_logFC': [row['mean_logFC']]}

    for node in node_dict:
        node_dict[node]['mean_logFC'] = sum(node_dict[node]['mean_logFC']) / len(node_dict[node]['mean_logFC'])

    edge_dict = {}
    for index, row in niche_df.iterrows():
        if pd.isna(row['patient_ids']).any():
            continue
        nodes = row[annotation_key].split('__')
        edges = combinations(nodes, 2)
        for edge in edges:
            edge = tuple(sorted(edge)) #ensure the edge (node1, node2) is the same as (node2, node1)
            if edge in edge_dict:
                edge_dict[edge].update(row['patient_ids'])
            else:
                edge_dict[edge] = set(row['patient_ids']) 

    edge_weights = {edge: len(patients) for edge, patients in edge_dict.items()}

    G = nx.Graph()
    for node in node_dict:
        color = colors_dict.get(lineage_dict.get(node, 'default'), 'grey')
        G.add_node(node, color=color)

    for edge, weight in edge_weights.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    for u, v, data in G.edges(data=True):
        data['inv_weight'] = 1 / data['weight'] if data['weight'] != 0 else float('inf')

    return G

def compute_niche_network_centrality(G):
    """Computes centrality information for niche network graph

    Parameters
    ----------
    G: nx.Graph (default = None)
        niche network graph

    Returns
    ----------
    niche_df: pd.DataFrame
        dataframe containing betweenness and eigenvector centrality
    """
    eigen_cent = nx.eigenvector_centrality(G, weight = 'weight')
    eigen_cent = pd.DataFrame(eigen_cent, index = ['value']).transpose()
    eigen_cent['centrality'] = 'eigenvector'
    between_cent = nx.betweenness_centrality(G,weight='inv_weight')
    between_cent = pd.DataFrame(between_cent, index = ['value']).transpose()
    between_cent['centrality'] = 'betweenness'
    centrality_df = pd.concat([eigen_cent, between_cent], axis = 0)
    
    return centrality_df