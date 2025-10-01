import os
import numpy as np
import pandas as pd
import seaborn as sns
import quiche as qu
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.colors as colors
from matplotlib.colorbar import ColorbarBase
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge, FancyArrowPatch
from collections import Counter
import pathlib
from typing import List, Union, Optional, Dict, Tuple
from tqdm.auto import tqdm
from skimage import io
from skimage.segmentation import find_boundaries
from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import scanpy as sc
import anndata
sns.set_style('ticks')

def generate_colors(cmap: str = "viridis",
                    n_colors: int = 3,
                    alpha: float = 0.4):
    """Generate colors from matplotlib colormap"""
    if not isinstance(n_colors, int) or (n_colors < 2) or (n_colors > 6):
        raise ValueError("n_colors must be an integer between 2 and 6")
    if isinstance(cmap, list):
        colors = [scalar_mappable.to_rgba(color, alpha=alpha) for color in cmap]
    else:
        scalar_mappable = ScalarMappable(cmap=cmap)
        colors = scalar_mappable.to_rgba(range(n_colors), alpha=alpha).tolist()

    return colors[:n_colors]

def plot_niches(quiche_op,
                niche: str,
                fovs: List[str],
                segmentation_directory: Union[pathlib.Path, str],
                save_directory: Union[pathlib.Path, str],
                labels_key: str = 'cell_cluster',
                fov_key: str = "fov",
                segmentation_label_key: str = "label",
                annotation_key: str = "quiche_niche_neighborhood",
                seg_suffix: str = "_whole_cell.tiff",
                colors_dict: Optional[Dict[str, str]] = None,
                cmap: str = 'Set3',
                background_color: np.ndarray = np.array([0.3, 0.3, 0.3, 1]),
                style: str = "seaborn-v0_8-paper",
                fig_file_type: str = "png",
                figsize: Tuple = (6, 6),
                dpi: int = 400):
    """
    Plots niches mapped to each each FOV in the cohort.
    
    Parameters
    ----------
    quiche_op:
        quiche class after fitting the model 
    niche: str
        string specifying niche of interest to plot
    fovs: List[str]
        list of FOVs to plot.
    segmentation_directory: Union[pathlib.Path, str]
        directory specifying where segmentation masks are stored. Must have the naming convention fov.tiff
    save_directory: Union[pathlib.Path, str]
        directory specifying where niche plots will be saved
    fov key: str (default = 'fov')
        column in mdata['spatial_nhood'].obs containing sample level information
    labels_key: str (default = 'cell_cluster')
        column in mdata['spatial_nhood'].obs containing cell phenotype information
    segmentation_label_key: str (default = 'label')
        column in mdata['spatial_nhood'].obs containing cell segmentation labels
    annotation_key: str (default = 'quiche_niche_neighborhood')
        column in mdata['spatial_nhood'].obs containing annotated quiche niches
    seg_suffix: str (default = '_whole_cell.tiff')
        filename suffix for segmentation images
    cmap: Union[str, np.ndarray] (default  = 'Set3')
        a matplotlib colormap name or a numpy array of RGBA colors for plotting
    colors_dict: Dict (default = None)
        dictonary mapping individual cell phenotypes to specific colors of interest. If None, will default to cmap
    background: np.ndarray, (default  = np.array([0.3, 0.3, 0.3, 1])
        RGBA value for cells that are not in the niche of interest
    style: str (default = 'seaborn-v0_8-paper')
        matplotlib style
    fig_file_type: str (default  = 'png')
        file extension for saved figures ('png', 'pdf')
    figsize: tuple (default  = (6, 6))
        size of the output figure
    dpi: int (default = 400)
        dots per inch for the saved figure

    Returns
    ----------
    None
    """
    plt.style.use(style)
    if isinstance(segmentation_directory, str):
        segmentation_directory = pathlib.Path(segmentation_directory)
    if save_directory is None:
       save_directory = pathlib.Path(os.path.join('figures', metric))
    elif isinstance(save_directory, str):
        save_directory = pathlib.Path(save_directory)
    if not save_directory.exists():
        save_directory.mkdir(parents=True, exist_ok=True)
    if isinstance(fovs, str):
        fovs = [fovs]

    for sub_dir in ["niche_masks", "niche_masks_colored", "niche_plots"]:
        (save_directory / sub_dir).mkdir(parents=True, exist_ok=True)

    if not hasattr(quiche_op, 'mdata'):
        raise AttributeError("Must run quiche first.")

    subset_mdata = quiche_op.mdata['spatial_nhood'][quiche_op.mdata['spatial_nhood'].obs.loc[:, annotation_key] == niche]
    cell_type_list = niche.split('__')
    df_cells = subset_mdata.to_df()
    df_cells[labels_key] = subset_mdata.obs[labels_key]
    df_cells[segmentation_label_key]= subset_mdata.obs[segmentation_label_key]
    df_cells[fov_key] = subset_mdata.obs[fov_key]
    df_cells = df_cells[np.isin(df_cells[labels_key], cell_type_list)]

    if colors_dict is None:
        if cmap is None:
            cmap = 'Set3'
        unique_clusters = sorted(df_cells[labels_key].unique())
        color_map = cm.get_cmap(cmap, len(unique_clusters))
        colors_dict = {}
        for idx, cluster in enumerate(unique_clusters):
            colors_dict[cluster] = color_map(idx)
    else:
        for cluster_name, color_val in colors_dict.items():
            if isinstance(color_val, str):
                colors_dict[cluster_name] = mcolors.to_rgba(color_val)

    with tqdm(total=len(fovs), desc="Plotting niches", unit="FOVs") as pbar:
        for fov in fovs:
            pbar.set_postfix(FOV=fov)

            seg_path = segmentation_directory / f"{fov}{seg_suffix}"
            if not seg_path.exists():
                raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
            
            seg_img = io.imread(seg_path)
            seg_img = np.squeeze(seg_img)
            seg_img[find_boundaries(seg_img, mode='inner')] = 0

            fov_data = df_cells[df_cells[fov_key] == fov].copy()

            label_cluster_map = dict(zip(fov_data[segmentation_label_key].astype(int), fov_data[labels_key]))
            colored_image = np.full((seg_img.shape[0], seg_img.shape[1], 4), background_color, dtype=np.float32)
            unique_cells = np.unique(seg_img)[1:] #0 is background
            for cell_id in unique_cells:
                cluster_name = label_cluster_map.get(cell_id, None)
                if cluster_name is not None:
                    if cluster_name in colors_dict:
                        colored_image[seg_img == cell_id] = colors_dict[cluster_name]
                    else:
                        pass

            black_mask = seg_img == 0
            colored_image[black_mask] = [0, 0, 0, 1]

            niche_mask_path = save_directory / "niche_masks" / f"{fov}_niche_mask.tiff"
            io.imsave(str(niche_mask_path), seg_img.astype(np.int32), check_contrast=False)

            niche_colored_path = save_directory / "niche_masks_colored" / f"{fov}_niche_colored.tiff"
            io.imsave(str(niche_colored_path), (colored_image * 255).astype(np.uint8), check_contrast=False)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.imshow(colored_image)
            ax.axis("off")
            
            all_clusters_used = sorted(list(colors_dict.keys()))
            
            cat_colors = [colors_dict[c] for c in all_clusters_used]
            discrete_cmap = colors.ListedColormap(cat_colors)
            
            bounds = np.arange(len(all_clusters_used) + 1) - 0.5
            discrete_norm = colors.BoundaryNorm(bounds, discrete_cmap.N)

            mappable = cm.ScalarMappable(norm=discrete_norm, cmap=discrete_cmap)
            mappable.set_array([])

            ##dynamically set cbar under the plot. not vertical for labeling 
            pos = ax.get_position()
            cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.03, pos.width, 0.02])
            cbar = plt.colorbar(mappable, cax=cbar_ax, orientation="horizontal", ticks=np.arange(len(all_clusters_used)))
            cbar.ax.set_xticklabels(all_clusters_used, rotation=0, ha="center")
            cbar.ax.tick_params(axis='x', length=0)

            cbar.outline.set_edgecolor("black")
            cbar.outline.set_linewidth(1)
            cbar.ax.tick_params(labelsize=10, length = 0)

            niche_plot_path = save_directory / "niche_plots" / f"{fov}.{fig_file_type}"
            fig.savefig(niche_plot_path, bbox_inches="tight")
            pbar.update(1)

def plot_niche_scores(quiche_op,
                      niche: str,
                      fovs: List[str],
                      segmentation_directory: Union[pathlib.Path, str],
                      save_directory: Union[pathlib.Path, str],
                      labels_key: str = 'cell_cluster',
                      fov_key: str = "fov",
                      segmentation_label_key: str = "label",
                      annotation_key: str = "quiche_niche_neighborhood",
                      metric: str = "SpatialFDR",
                      seg_suffix: str = "_whole_cell.tiff",
                      vmin: Optional[Union[int, float]] = None,
                      vmax: Optional[Union[int, float]] = None,
                      cmap: Union[str, np.ndarray] = "vlag",
                      background_color: np.ndarray = np.array([0.3, 0.3, 0.3, 1]),
                      style: str = "seaborn-v0_8-paper",
                      fig_file_type: str = "png",
                      figsize: Tuple = (6, 6),
                      dpi: int = 400):
    """
    Plots niches mapped to each each FOV in the cohort.
    
    Parameters
    ----------
    quiche_op:
        quiche class after fitting the model 
    niche: str
        string specifying niche of interest to plot
    fovs: List[str]
        list of FOVs to plot.
    segmentation_directory: Union[pathlib.Path, str]
        directory specifying where segmentation masks are stored. Must have the naming convention fov.tiff
    save_directory: Union[pathlib.Path, str]
        directory specifying where niche plots will be saved
    fov key: str (default = 'fov')
        column in mdata['spatial_nhood'].obs containing sample level information
    labels_key: str (default = 'cell_cluster')
        column in mdata['spatial_nhood'].obs containing cell phenotype information
    segmentation_label_key: str (default = 'label')
        column in mdata['spatial_nhood'].obs containing cell segmentation labels
    annotation_key: str (default = 'quiche_niche_neighborhood')
        column in mdata['spatial_nhood'].obs containing annotated quiche niches
    metric: str (default  = 'spatialFDR')
        column in mdata['quiche'].var containing metric of interest for plotting (e.g. 'SpatialFDR', 'logFC', '-log10(SpatialFDR)'
    seg_suffix: str (default = '_whole_cell.tiff')
        filename suffix for segmentation images
    vmin: float (default = 0)
        minimum value for colormap normalization
    vmax: float (default = 10)
        maximum value for colormap normalization
    cmap: Union[str, np.ndarray] (default  = 'vlag')
        a matplotlib colormap name or a numpy array of RGBA colors for plotting
    background: np.ndarray, (default  = np.array([0.3, 0.3, 0.3, 1])
        RGBA value for cells that are not in the niche of interest
    style: str (default = 'seaborn-v0_8-paper')
        matplotlib style
    fig_file_type: str (default  = 'png')
        file extension for saved figures ('png', 'pdf')
    figsize: tuple (default  = (6, 6))
        size of the output figure
    dpi: int (default = 400)
        dots per inch for the saved figure

    Returns
    ----------
    None
    """
    plt.style.use(style)
    if isinstance(segmentation_directory, str):
        segmentation_directory = pathlib.Path(segmentation_directory)
    if save_directory is None:
       save_directory = pathlib.Path(os.path.join('figures', metric))
    elif isinstance(save_directory, str):
        save_directory = pathlib.Path(save_directory)
    if not save_directory.exists():
        save_directory.mkdir(parents=True, exist_ok=True)
    if isinstance(fovs, str):
        fovs = [fovs]

    for sub_dir in ["niche_masks", "niche_masks_colored", "niche_plots"]:
        (save_directory / sub_dir).mkdir(parents=True, exist_ok=True)

    if not hasattr(quiche_op, 'mdata'):
        raise AttributeError("Must run quiche first.")
    
    try:
        quiche_op.mdata['spatial_nhood'].obs[metric] = quiche_op.mdata['quiche'].var[metric].values
    except:
        raise KeyError(f"{metric} is not a valid metric.")

    subset_mdata = quiche_op.mdata['spatial_nhood'][quiche_op.mdata['spatial_nhood'].obs.loc[:, annotation_key] == niche]
    cell_type_list = niche.split('__')
    df_cells = subset_mdata.to_df()
    df_cells[labels_key] = subset_mdata.obs[labels_key]
    df_cells[segmentation_label_key]= subset_mdata.obs[segmentation_label_key]
    df_cells[fov_key] = subset_mdata.obs[fov_key]
    df_cells[metric] = subset_mdata.obs[metric].values
    df_cells = df_cells[np.isin(df_cells[labels_key], cell_type_list)]

    if isinstance(cmap, str):
        color_map = cm.get_cmap(cmap)
    else:
        color_map = colors.ListedColormap(cmap)

    if vmin is None:
        vmin = df_cells[metric].min()
    if vmax is None:
        vmax = df_cells[metric].max()

    norm = Normalize(vmin=vmin, vmax=vmax)

    with tqdm(total=len(fovs), desc="Plotting niche scores", unit="FOVs") as pbar:
        for fov in fovs:
            pbar.set_postfix(FOV=fov)

            seg_path = segmentation_directory / f"{fov}{seg_suffix}"
            if not seg_path.exists():
                raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
            
            seg_img = io.imread(seg_path)
            seg_img = np.squeeze(seg_img)
            seg_img[find_boundaries(seg_img, mode='inner')] = 0

            fov_data = df_cells[df_cells[fov_key] == fov].copy()

            label_metric_map = dict(zip(fov_data[segmentation_label_key].astype(int), fov_data[metric]))

            scores = np.full(seg_img.shape, np.nan, dtype=np.float32)
            unique_cells = np.unique(seg_img)[1:]  # skip 0 (background)
            for cell_id in unique_cells:
                if cell_id in label_metric_map and not pd.isnull(label_metric_map[cell_id]):
                    scores[seg_img == cell_id] = label_metric_map[cell_id]
                else:
                    scores[seg_img == cell_id] = np.nan

            colored_image = color_map(norm(scores))

            nan_mask = np.isnan(scores)
            colored_image[nan_mask] = background_color

            black_mask = seg_img == 0
            colored_image[black_mask] = [0, 0, 0, 1]

            niche_mask_path = save_directory / "niche_masks" / f"{fov}_niche_mask.tiff"
            io.imsave(str(niche_mask_path), scores, check_contrast=False)

            niche_colored_path = save_directory / "niche_masks_colored" / f"{fov}_niche_colored.tiff"
            io.imsave(str(niche_colored_path), (colored_image * 255).astype(np.uint8), check_contrast=False)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.imshow(colored_image)
            ax.axis("off")

            #dynamically create nice cbar 
            pos = ax.get_position()
            cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])
            mappable = cm.ScalarMappable(norm=norm, cmap=color_map)
            cbar = plt.colorbar(mappable, cax=cbar_ax)
            cbar.set_label(metric, fontsize=12, labelpad=10)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            cbar.ax.tick_params(labelsize=10)
            cbar.outline.set_edgecolor('black')
            cbar.outline.set_linewidth(1)
            
            niche_plot_path = save_directory / "niche_plots" / f"{fov}.{fig_file_type}"
            fig.savefig(niche_plot_path, bbox_inches='tight')
            pbar.update(1)

def beeswarm(quiche_op,
            alpha: float = 0.05,
            niches: Optional[List[str]] = None,
            figsize: Tuple = (6, 8),
            annotation_key: str = 'quiche_niche_neighborhood',
            condition_key: str = 'condition',
            logfc_key: str = 'logFC',
            pvalue_key: str = 'SpatialFDR',
            condition_legend_loc: Union[List[int], List[float]] = [1, 0.9],
            xlim: Union[List[int], List[float]] = [-5, 5],
            fontsize: int = 8,
            colors_dict: Dict = {'0':'#377eb8', '1':'#e41a1c'},
            save_directory: Optional[Union[pathlib.Path, str]] = None,
            filename_save: Optional[str] = None):
    """
    Plot a beeswarm plot of logFC for niche neighborhoods without barplots of binary metadata information.
    Modified from pertpy: https://pertpy.readthedocs.io/en/latest/usage/tools/pertpy.tools.Milo.html
    
    Parameters
    ----------
    quiche_op:
        quiche class after fitting the model 
    alpha: float (default = 0.05)
        alpha level to consider signficiance 
    niches: List
        list of niches to plot
    save_directory: Union[pathlib.Path, str] or None
        directory specifying where beeswarm plots will be saved
    annotation_key: str (default = 'quiche_niche_neighborhood')
        column in mdata['quiche'].obs containing annotated quiche niches
    condition_key: str (default = 'condition')
        column in mdata['quiche'].obs containing condition-level information
    logfc_key: str (default  = 'logFC')
        column in mdata['quiche'].var containing log fold change metric of interest for plotting
    pvalue_key: str (default  = 'SpatialFDR')
        column in mdata['quiche'].var containing significance metric of interest for plotting
    condition_legend_loc: Union[List[int], List[float]] (default = [1, 0.9])
        xy location on plot for condition legend
    xlim: Union[List[int], List[float]] (default = [-5, 5])
        x-axis limits on beeswarm plot
    fontsize: int (default  = 8)
        fontsize for plot
    background: np.ndarray, (default  = np.array([0.3, 0.3, 0.3, 1])
        RGBA value for cells that are not in the niche of interest
    colors_dict: Dict (default = {'0':'#377eb8', '1':'#e41a1c'})
        dictionary specifying colors for condition labels
    save_directory: Optional[Union[pathlib.Path, str]] (default = None)
        string specifying where plots should be saved. If None, will not save
    filename_save: Optional[str] (default = None)
        string specifying filename for saving. If None and save_directory is specified, will save as 'beeswarm'

    Returns
    ----------
    None
    """
    sns.set_style('ticks')
    if not hasattr(quiche_op, 'mdata'):
        raise AttributeError("Must run quiche first.")

    mdata = quiche_op.mdata

    sns.set_style('ticks')
    nhood_adata = mdata['quiche'].T.copy()

    try:
        nhood_adata.obs[annotation_key]
    except KeyError:
        raise RuntimeError(f"'{annotation_key}' not defined in nhood_adata.obs.")

    try:
        nhood_adata = nhood_adata[np.isin(nhood_adata.obs[annotation_key], niches)]  # Subsets data by niches of interest
    except:
        raise RuntimeError('Specify a list of niches to plot.')

    try:
        nhood_adata.obs[logfc_key]
    except KeyError:
        raise RuntimeError(f"'{logfc_key}' not found in mdata['quiche'].obs.")

    try:
        nhood_adata.obs[pvalue_key]
    except KeyError:
        raise RuntimeError(f"'{pvalue_key}' not found in mdata['quiche'].obs.")

    sorted_annos = (nhood_adata.obs[[annotation_key, logfc_key]].groupby(annotation_key).mean().sort_values(logfc_key, ascending=True).index)
    anno_df = nhood_adata.obs[[annotation_key, logfc_key, pvalue_key]].copy()
    anno_df["is_signif"] = anno_df[pvalue_key] < alpha
    anno_df = anno_df[anno_df[annotation_key] != "nan"]

    cmap_df = pd.DataFrame(mdata['quiche'].var.groupby(annotation_key)[logfc_key].mean(), columns=[logfc_key])
    cmap = np.full(np.shape(mdata['quiche'].var.groupby(annotation_key)[logfc_key].mean())[0], 'lightgrey', dtype='object')
    cmap[mdata['quiche'].var.groupby(annotation_key)[logfc_key].mean() <= 0] = list(colors_dict.values())[0]
    cmap[mdata['quiche'].var.groupby(annotation_key)[logfc_key].mean() > 0] = list(colors_dict.values())[1]

    cmap_df['cmap'] = cmap

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=anno_df,
        y=annotation_key,
        x=logfc_key,
        order=sorted_annos,
        inner=None,
        orient="h",
        palette=cmap_df.loc[sorted_annos]['cmap'].values,
        linewidth=0,
        scale="width",
        alpha=0.8,
        ax=ax
    )
    sns.stripplot(
        data=anno_df,
        y=annotation_key,
        x=logfc_key,
        order=sorted_annos,
        hue="is_signif",
        palette={False: "lightgrey", True: "black"},
        size=2,
        orient="h",
        alpha=0.7,
        ax=ax,
    )

    ax.set_xlabel('log2(fold change)', fontsize=fontsize)
    ax.set_ylabel(annotation_key, fontsize=fontsize)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)

    handles, labels = ax.get_legend_handles_labels()
    if "is_signif" in labels:
        signif_index = labels.index("is_signif")
        handles.pop(signif_index)
        labels.pop(signif_index)

    fdr_legend = ax.legend(
        handles,
        labels,
        title=f"SpatialFDR \n (α < {alpha})",
        loc='upper right',
        fontsize=fontsize,
        title_fontsize=fontsize,
        markerscale=2,
        frameon=False
    )

    condition_patches = [Patch(color=colors_dict[key], label=key) for key in colors_dict]

    condition_legend = ax.legend(
        condition_patches,
        list(colors_dict.keys()),
        title=condition_key.replace('_', ' ').title(),
        loc='upper right',
        bbox_to_anchor=(condition_legend_loc[0], condition_legend_loc[1]),
        fontsize=fontsize,
        markerscale=2,
        title_fontsize=fontsize,
        frameon=False
    )

    ax.add_artist(fdr_legend)
    ax.add_artist(condition_legend)

    ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        if filename_save is not None:
            plt.savefig(os.path.join(save_directory, f'{filename_save}.pdf'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_directory, f'beeswarm.pdf'), bbox_inches='tight')
    else:
        plt.show()

def beeswarm_proportion(quiche_op,
                        niche_metadata: Optional[pd.DataFrame] = None,
                        alpha: float = 0.05,
                        niches: Optional[List[str]] = None,
                        figsize: Tuple = (6, 8),
                        annotation_key: str = 'quiche_niche_neighborhood',
                        condition_key: str = 'condition',
                        patient_key: Optional[str] = None,
                        logfc_key: str = 'logFC',
                        pvalue_key: str = 'SpatialFDR',
                        xlim: Union[List[int], List[float]] = [-5, 5],
                        condition_legend_loc: Union[List[int], List[float]] = [1, 0.9],
                        fontsize: int = 8,
                        xlim_proportion: Union[List[int], List[float]] = [-1, 1],
                        colors_dict: Dict = {'0':'#377eb8', '1':'#e41a1c'},
                        save_directory: Optional[Union[pathlib.Path, str]] = None,
                        filename_save: Optional[str] = None):
        """
        Plot a beeswarm plot of logFC for niche neighborhoods without patient metadata information.
        Modified from pertpy: https://pertpy.readthedocs.io/en/latest/usage/tools/pertpy.tools.Milo.html
        
        Parameters
        ----------
        quiche_op:
            quiche class after fitting the model 
        niche_metadata: pd.DataFrame (default = None)
            pandas dataframe specifying patient-level information. Can be accessed by running qu.tl.compute_niche_metadata. If None, will run with defaults
        alpha: float (default = 0.05)
            alpha level to consider signficiance 
        niches: List
            list of niches to plot
        save_directory: Union[pathlib.Path, str] or None
            directory specifying where beeswarm plots will be saved
        annotation_key: str (default = 'quiche_niche_neighborhood')
            column in mdata['quiche'].obs containing annotated quiche niches
        condition_key: str (default = 'condition')
            column in mdata['quiche'].obs containing condition-level information
        patient_key: Optional[str] (default = None)
            column in mdata['quiche'].obs containing patient-level information. Only used if niche_metadata is unspecified 
        logfc_key: str (default  = 'logFC')
            column in mdata['quiche'].var containing log fold change metric of interest for plotting
        pvalue_key: str (default  = 'SpatialFDR')
            column in mdata['quiche'].var containing significance metric of interest for plotting
        condition_legend_loc: Union[List[int], List[float]] (default = [1, 0.9])
            xy location on plot for condition legend
        xlim: Union[List[int], List[float]] (default = [-5, 5])
            x-axis limits on beeswarm plot
        xlim_proportion: Union[List[int], List[float]] (default = [-1, 1])
            x-axis limits on barplots
        fontsize: int (default  = 8)
            fontsize for plot
        background: np.ndarray, (default  = np.array([0.3, 0.3, 0.3, 1])
            RGBA value for cells that are not in the niche of interest
        colors_dict: Dict (default = {'0':'#377eb8', '1':'#e41a1c'})
            dictionary specifying colors for condition labels
        save_directory: Optional[Union[pathlib.Path, str]] (default = None)
            string specifying where plots should be saved. If None, will not save
        filename_save: Optional[str] (default = None)
            string specifying filename for saving. If None and save_directory is specified, will save as 'beeswarm'

        Returns
        ----------
        None

        Example usage
        ----------
        qu.pl.beeswarm_proportion(quiche_op,
                                niche_metadata = niche_metadata,
                                niches = niches,
                                xlim = [-3,3],
                                xlim_proportion = [-0.3, 0.3],
                                logfc_key = 'logFC',
                                pvalue_key = 'SpatialFDR',
                                annotation_key = 'quiche_niche_neighborhood',
                                condition_key = 'Relapse',
                                figsize = (6, 12),
                                fontsize = 10,
                                colors_dict = {'0': '#377eb8', '1': '#e41a1c'},
                                save_directory = None,
                                filename_save = None)
        """
        sns.set_style('ticks')
        if not hasattr(quiche_op, 'mdata'):
            raise AttributeError("Must run quiche first.")
        
        mdata = quiche_op.mdata
        nhood_adata = mdata['quiche'].T.copy()

        try:
            nhood_adata.obs[annotation_key]
        except KeyError:
            raise RuntimeError(f"{annotation_key} not defined")

        try:
            if niche_metadata is None:
                niche_metadata = qu.tl.compute_niche_metadata(mdata,
                                                niches = niches,
                                                annotation_key =annotation_key,
                                                patient_key = patient_key,
                                                condition_key = condition_key,
                                                niche_threshold = 0,
                                                condition_type  = 'binary',
                                                metrics = ['logFC'])
            
            niches = list(set(niches).intersection(set(niche_metadata[annotation_key].unique()))) #ensures niches are in metadata and pass niche_threshold
            nhood_adata = nhood_adata[np.isin(nhood_adata.obs[annotation_key], niches)] #subsets data by niches of interest
        except:
             raise RuntimeError(f'Specify a list of niches to plot.')
        
        try:
            nhood_adata.obs[logfc_key]
        except KeyError:
            raise RuntimeError(f"'{logfc_key}' not in mdata.uns['nhood_adata'].obs.")
        
        try:
            nhood_adata.obs[pvalue_key]
        except KeyError:
            raise RuntimeError(f"'{pvalue_key}' not in mdata.uns['nhood_adata'].obs.")
        
        sorted_annos = (nhood_adata.obs[[annotation_key, logfc_key]].groupby(annotation_key).mean().sort_values(logfc_key, ascending=True).index)

        anno_df = nhood_adata.obs[[annotation_key, logfc_key, pvalue_key]].copy()
        anno_df["is_signif"] = anno_df[pvalue_key] < alpha
        anno_df = anno_df[anno_df[annotation_key] != "nan"]

        cmap_df = pd.DataFrame(mdata['quiche'].var.groupby(annotation_key)[logfc_key].mean(), columns = [logfc_key])
        cmap = np.full(np.shape(mdata['quiche'].var.groupby(annotation_key)[logfc_key].mean())[0], 'lightgrey', dtype = 'object')
        cmap[mdata['quiche'].var.groupby(annotation_key)[logfc_key].mean() <= 0] = list(colors_dict.values())[0]
        cmap[mdata['quiche'].var.groupby(annotation_key)[logfc_key].mean() > 0] = list(colors_dict.values())[1]

        cmap_df['cmap'] = cmap
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[1, 0.4])  # 2 columns with equal width

        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        
        g = sns.violinplot(
                data=anno_df,
                y=annotation_key,
                x=logfc_key,
                order=sorted_annos,
                inner=None,
                orient="h",
                palette= cmap_df.loc[sorted_annos]['cmap'].values,
                linewidth=0,
                scale="width",
                alpha = 0.8,
                ax=ax0
            )

        g = sns.stripplot(
                data=anno_df,
                y=annotation_key,
                x=logfc_key,
                order=sorted_annos,
                hue="is_signif",
                palette={False: "lightgrey", True: "black"},
                size=2,
                orient="h",
                alpha=0.7,
                ax = ax0,
            )

        ax0.set_xlabel('log2(fold change)', fontsize = fontsize)
        ax0.set_ylabel(annotation_key, fontsize = fontsize)
            
        df_pivot = niche_metadata.pivot_table(index=annotation_key, columns=condition_key, values='proportion_patients_niche')
        df_pivot = df_pivot.loc[niche_metadata[[annotation_key, 'med_logFC']].sort_values(by='med_logFC', ascending=False).drop_duplicates()[annotation_key].values]
        min_name = mdata['quiche'].var.groupby(condition_key)[logfc_key].mean().idxmin()
        max_name = mdata['quiche'].var.groupby(condition_key)[logfc_key].mean().idxmax()
        df_pivot.loc[:, min_name] = df_pivot.loc[:, min_name] * -1
        df_pivot = df_pivot.loc[sorted_annos, :]
        handles, labels = g.get_legend_handles_labels()

        ax1.barh(df_pivot.index, df_pivot.loc[:, min_name],  height = 0.5, color = colors_dict[min_name], edgecolor = 'none', label = df_pivot.iloc[:, 0].name)
        ax1.barh(df_pivot.index, df_pivot.loc[:, max_name], height = 0.5, color = colors_dict[max_name], edgecolor = 'none', left=0, label = df_pivot.iloc[:, 1].name)
        ax1.set_xlim(xlim_proportion[0], xlim_proportion[1])

        ax1.set_yticks(ax0.get_yticks())
        ax1.set_yticklabels(ax0.get_yticklabels())
        ax1.set_ylim(ax0.get_ylim())
        ax1.set_yticklabels([])
        ax1.set_yticks([])

        ax1.set_xlabel('proportion', fontsize=fontsize)
        ax1.set_ylabel('')
        ax0.tick_params(labelsize=fontsize)
        ax1.tick_params(labelsize=fontsize)

        xticks = ax1.get_xticks()
        ax1.set_xticklabels([str(abs(np.round(x, 2))) for x in xticks])

        if xlim is not None:
            ax0.set_xlim(xlim[0], xlim[1])

        ax0.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--", linewidth = 1)
        ax1.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--", linewidth = 1)

        condition_patches = [Patch(color=colors_dict[min_name], label=min_name), Patch(color=colors_dict[max_name], label=max_name)]

        fdr_legend = ax0.legend(
            handles,
            labels,
            title=f"SpatialFDR \n (α < {alpha})",
            loc='upper right',
            fontsize=fontsize,
            title_fontsize=fontsize,
            markerscale = 2,
            frameon=False
        )

        condition_legend = ax0.legend(
            condition_patches,
            [min_name, max_name],
            title=condition_key.replace('_', ' ').title(),
            loc='upper right',
            bbox_to_anchor=(condition_legend_loc[0], condition_legend_loc[1]),
            fontsize=fontsize,
            markerscale=2,
            title_fontsize=fontsize,
            frameon=False
        )

        ax0.add_artist(fdr_legend)
        ax0.add_artist(condition_legend)
        if save_directory is not None:
            os.makedirs(save_directory, exist_ok=True)
            if filename_save is not None:
                plt.savefig(os.path.join(save_directory, f'{filename_save}.pdf'), bbox_inches='tight')
            else:
                plt.savefig(os.path.join(save_directory, f'beeswarm.pdf'), bbox_inches='tight')
        else:
            plt.show()

def plot_niche_network_donut(G: nx.Graph,
                            figsize: Tuple[float, float] = (6, 6),
                            save_directory: Optional[Union[str, os.PathLike]] = None,
                            filename_save: Optional[str] = None,
                            node_order: Optional[List[str]] = None,
                            fontsize: int = 10,
                            buffer: float = 1.5,
                            weightscale: float = 0.4,
                            min_node_size: float = 50,
                            max_node_size: float = 500,
                            donut_radius_inner: float = 1.15,
                            donut_radius_outer: float = 1.25,
                            centrality_measure: str = 'degree',
                            colors_dict: Optional[Dict[str, str]] = None,
                            lineage_dict: Optional[Dict[str, str]] = None,
                            curvature: float = 0.2,
                            edge_cmap: str = 'viridis',
                            vmin: Optional[float] = 1,
                            vmax: Optional[float] = None,
                            edge_width: float = 2,
                            label_lineages: bool = True,
                            angle_tolerance: float = 5,
                            edge_label: str = 'Patients'):
    """
    Plot niche network diagram using a circular layout.

    Parameters
    ----------
    G: nx.Graph
        networkX graph containing nodes, edges, and metadata information. Can obtain by running qu.tl.compute_niche_network
    figsize: tuple (default = (6, 6))
        figure size for plot
    save_directory: Optional[Union[pathlib.Path, str]] (default = None)
        string specifying where plots should be saved. If None, will not save
    filename_save: Optional[str] (default = None)
        string specifying filename for saving. If None and save_directory is specified, will save as 'niche_network'
    node_order: list of str or None (default = None)
        list specifying node order, should be ordered by lineage information. If None, will be placed according to G.nodes()
    fontsize: int (default = 10)
        font size for node labels
    buffer: float (default = 1.5)
        margin around the figure in plot coordinates
    weightscale: float (default = 0.4)
        scaling factor to adjust edge weights. edge weights are used to color the edges
    min_node_size: float (default = 50)
        minimum node size for scaling centrality
    max_node_size: float (default = 500)
        maximum node size for scaling centrality
    donut_radius_inner: float (default = 1.15)
        inner radius for the donut representing lineages
    donut_radius_outer: float (default = 1.25)
        outer radius for the donut representing lineages
    centrality_measure: str (default = 'degree')
        centrality measure for node sizing. (e.g. 'degree', 'betweenness', 'closeness', 'eigenvector')
    colors_dict: dict or None (default = None)
        dictionary mapping lineages to colors
    lineage_dict: dict or None (default = None)
        dictionary mapping nodes to lineages
    curvature: float (default = 0.2)
        curvature of the edges
    edge_cmap: str or matplotlib.cm.ScalarMappable (default = 'viridis')
        colormap name or object to color edges by their weight
    vmin: float (default = 1)
        minimum value for edge weight normalization
    vmax: float or None (default = None)
        maximum value for edge weight normalization. If None, will directly compute the max
    edge_width: float (default = 2)
        edge width
    label_lineages: bool (default = True)
        boolean specifying whether lineages should be labeled around the donut
    angle_tolerance: float (default = 5)
        degrees of tolerance for adjusting label angles near 90 or 270 degrees
    edge_label: str (default = 'Patients')
        colorbar label indicating what the edge weights represent

    Returns
    ----------
    None

    Example usage
    ----------
    colors_dict = {'myeloid':'#4DCCBD', 'lymphoid':'#279AF1', 'tumor':'#FF8484', 'structural':'#F9DC5C'}

    lineage_dict = {'APC':'myeloid', 'B':'lymphoid', 'CAF': 'structural', 'CD4T': 'lymphoid', 'CD8T': 'lymphoid',
                    'CD68_Mac': 'myeloid', 'CD163_Mac': 'myeloid', 'Cancer_1': 'tumor', 'Cancer_2': 'tumor', 'Cancer_3': 'tumor',
                    'Endothelium':'structural', 'Fibroblast': 'structural', 'Mac_Other': 'myeloid', 'Mast':'myeloid', 'Monocyte':'myeloid',
                    'NK':'lymphoid', 'Neutrophil':'myeloid', 'Smooth_Muscle':'structural', 'T_Other':'lymphoid', 'Treg':'lymphoid'}

    cell_ordering = ['Cancer_1', 'Cancer_2', 'Cancer_3', 'CD4T', 'CD8T', 'Treg', 'T_Other', 'B', 
                    'NK', 'CD68_Mac', 'CD163_Mac', 'Mac_Other', 'Monocyte', 'APC','Mast', 'Neutrophil',
                    'CAF', 'Fibroblast', 'Smooth_Muscle', 'Endothelium']

    ## niche network for patients that did not relapse, ie logFC < 0

    niche_metadata_neg = niche_metadata[(niche_metadata['mean_logFC'] < 0) & (niche_metadata['n_patients_niche'] > 1)]

    G1 = qu.tl.compute_niche_network(niche_df = niche_metadata_neg,
                            colors_dict = colors_dict,
                            lineage_dict = lineage_dict,
                            annotation_key = 'quiche_niche_neighborhood')

    qu.pl.plot_niche_network_donut(G = G1,
                                figsize=(6, 6),
                                node_order = cell_ordering,
                                centrality_measure = 'eigenvector',
                                colors_dict = colors_dict,
                                lineage_dict=lineage_dict, 
                                donut_radius_inner = 1.15,
                                donut_radius_outer = 1.25,
                                edge_cmap = 'bone_r',
                                edge_label = 'Patients',
                                save_directory=None,
                                filename_save=None)
    """
    try:
        edge_cmap = cm.get_cmap(edge_cmap)
    except:
        raise ValueError(f"{edge_cmap} is not a valid cmap.")

    if node_order is None:
        node_order = list(G.nodes())
        label_lineages = False #if nodes are not labeled in order, then lineage labels won't make sense
    
    if (lineage_dict is not None) & (colors_dict is None):
        raise ValueError("Specify colors_dict.")

    for node in node_order:
        if node not in G.nodes:
            G.add_node(node, lineage='Unknown', color='lightgrey')

    if lineage_dict is not None:
        for node in G.nodes():
            lineage = lineage_dict.get(node, 'Unknown')
            G.nodes[node]['lineage'] = lineage

    num_nodes = len(node_order)
    pos = {node: [np.cos(2 * np.pi * i / num_nodes), np.sin(2 * np.pi * i / num_nodes)] 
           for i, node in enumerate(node_order)}

    if vmax is None:
        vmax = np.max([G[u][v].get('weight', 1.0) for u, v in G.edges()])

    norm = Normalize(vmin = vmin, vmax = vmax)

    if centrality_measure == 'degree':
        centrality = {node: sum(G[node][neighbor].get('weight', 1.0) for neighbor in G.neighbors(node)) 
                      for node in G.nodes()}
    elif centrality_measure == 'betweenness':
        centrality = nx.betweenness_centrality(G, weight='weight')
    elif centrality_measure == 'closeness':
        centrality = nx.closeness_centrality(G, distance='weight')
    elif centrality_measure == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
        except nx.NetworkXException as e:
            raise ValueError(f"Error computing eigenvector centrality: {e}")
    else:
        raise ValueError(f"Unsupported centrality measure: {centrality_measure}")

    centrality_values = list(centrality.values())
    min_centrality = min(centrality_values)
    max_centrality = max(centrality_values)

    #calculate node sizes based on centrality
    node_sizes = [
        min_node_size + (centrality[node] - min_centrality) / (max_centrality - min_centrality) * 
        (max_node_size - min_node_size) if centrality[node] >= 0.001 else 0
        for node in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=figsize)
    for (u, v, data) in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        color = edge_cmap(norm(data.get('weight', 1.0) * weightscale / weightscale)) #color edges according to weight
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={curvature}",
            color=color,
            linewidth=edge_width,
            antialiased=True,
            arrowstyle='-',
        )
        ax.add_patch(arrow)

    nx.draw_networkx_nodes(
        G, pos, 
        node_color=[G.nodes[node].get('color', 'lightgrey') for node in G.nodes()], 
        node_size=node_sizes, 
        alpha=0.9, 
        edgecolors='black',
        linewidths=0.5,
        ax=ax
    )

    for node, (x, y) in pos.items():
        label_color = '#B2B0BF' if centrality[node] < 0.001 else 'black'
        angle_rad = np.arctan2(y, x)
        angle_deg = (np.degrees(angle_rad)) % 360
        is_near_90 = (90 - angle_tolerance) <= angle_deg <= (90 + angle_tolerance)
        is_near_270 = (270 - angle_tolerance) <= angle_deg <= (270 + angle_tolerance)
        offset_distance = donut_radius_outer + 0.05
        offset_x = offset_distance * np.cos(angle_rad)
        offset_y = offset_distance * np.sin(angle_rad)
        if is_near_90:
            ha = 'left'
            va = 'center'
            rotation_angle = 90
        elif is_near_270:
            ha = 'left'
            va = 'bottom'
            rotation_angle = 270 
        else:
            ha = 'left' if (angle_deg < 90 or angle_deg > 270) else 'right'
            va = 'bottom' if y < 0 else 'top'
            if 90 < angle_deg < 270:
                rotation_angle = (angle_deg + 180) % 360
            else:
                rotation_angle = angle_deg
        
        ax.text(offset_x, offset_y, node,
            fontsize=fontsize, ha=ha, va=va,
            color=label_color,
            rotation=rotation_angle, rotation_mode='anchor')

    ax.set_xlim(-1 * buffer, 1 * buffer)
    ax.set_ylim(-1 * buffer, 1 * buffer)
    ax.set_aspect('equal')
    ax.axis('off')

    if lineage_dict is not None:
        lineage_counts = Counter([G.nodes[node]['lineage'] for node in node_order])
        unique_lineages = list(lineage_counts.keys())
        lineage_colors_donut = {
            lineage: colors_dict.get(lineage, 'grey') if colors_dict else edge_cmap(i / len(unique_lineages))
            for i, lineage in enumerate(unique_lineages)
        }

        angles_per_node = {node: 360 * i / num_nodes for i, node in enumerate(node_order)}
        for node in node_order:
            lineage = G.nodes[node].get('lineage', 'Unknown')
            angle = angles_per_node[node]
            theta1 = angle - (360 / num_nodes) / 2
            theta2 = angle + (360 / num_nodes) / 2
            wedge = Wedge(
                center=(0, 0),
                r=donut_radius_outer,
                theta1=theta1,
                theta2=theta2,
                width=donut_radius_outer - donut_radius_inner,
                facecolor=lineage_colors_donut[lineage],
                edgecolor='none'
            )
            ax.add_patch(wedge)
        
        if label_lineages == True:
            lineage_angles = {}
            for lineage in unique_lineages:
                lineage_node_angles = [angles_per_node[node] for node in node_order if G.nodes[node]['lineage'] == lineage]
                mean_angle = np.mean(lineage_node_angles) % 360
                lineage_angles[lineage] = mean_angle

            for lineage, angle in lineage_angles.items():
                angle_rad = np.deg2rad(angle)
                label_x = (donut_radius_inner + donut_radius_outer) / 2 * np.cos(angle_rad)
                label_y = (donut_radius_inner + donut_radius_outer) / 2 * np.sin(angle_rad)

                rotation_angle = angle + 90
                if 90 < rotation_angle <= 270:
                    rotation_angle -= 180

                ax.text(
                    label_x, label_y, lineage,
                    ha='center', va='center', fontsize=fontsize,
                    rotation=rotation_angle, rotation_mode='anchor', color='white'
                )

    cbar_ax = fig.add_axes([0.82, 0.1, 0.12, 0.02])
    cbar = ColorbarBase(cbar_ax, cmap=edge_cmap, norm=norm, orientation='horizontal')
    cbar.set_label(edge_label, fontsize=fontsize)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{int(vmin)}", f"{int(vmax)}"])
    cbar.ax.tick_params(labelsize=fontsize-2)

    plt.tight_layout()
    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        if filename_save is not None:
            plt.savefig(os.path.join(save_directory, filename_save + '.pdf'), bbox_inches='tight', dpi=400)
        else:
            plt.savefig(os.path.join(save_directory,  'niche_network.pdf'), bbox_inches='tight', dpi=400)
    else:
        plt.show()

def plot_differential_expression(quiche_op,
                                niches: Optional[List[str]] = None, 
                                annotation_key: str = 'quiche_niche_neighborhood',
                                labels_key: str = 'cell_cluster',
                                markers: Optional[List[str]] = None,
                                figsize: Tuple[float, float] = (6, 4.5),
                                cmap: str = 'vlag',
                                vmin: float = -1,
                                vmax: float = 1,
                                vcenter: float = 0,
                                dendrogram: bool = True,
                                save_directory: Optional[Union[str, os.PathLike]] = None,
                                filename_save: Optional[str] = None):
    """
    Plot the change in standardized expression of cell types within niches as compare to the entire cohort.

    Parameters
    ----------
    quiche_op:
        quiche class after fitting the model and running functional expression
    niches: list of str or None (default = None)
        list of niches for plotting
    annotation_key: str (default = 'quiche_niche_neighborhood')
        column in quiche_op.adata_func.obs specifying annotated niches
    labels_key: str (default = 'cell_cluster')
        column in quiche_op.adata_func.obs specifying cell phenotype information
    markers: list or None (default = None)
        list of functional markers (genes/proteins/features) to plot. if None, will plot all
    figsize: tuple (default = (6, 4.5))
        figure size
    cmap: str (default = 'vlag')
        heatmap colormap
    vmin: float (default = -1)
        minimum value for colormap normalization
    vmax: float (default = 1)
        maximum value for colormap normalization
    vcenter: float (default = 0)
        center value for colormap normalization
    dendrogram: bool (default = True)
        boolean specifying whether to include a dendrogram 
    save_directory: Optional[Union[pathlib.Path, str]] (default = None)
        string specifying where plots should be saved. If None, will not save
    filename_save: Optional[str] (default = None)
        string specifying filename for saving. If None and save_directory is specified, will save as 'matrixplot'

    Returns
    ----------
    None
    """    
    if not hasattr(quiche_op, 'adata_func'):
        raise AttributeError("Must run quiche_op.compute_functional_expression first.")
    
    if markers is None:
        markers = quiche_op.mdata['expression'].var_names
    
    if niches is None:
        raise ValueError(f"Specify a list of niches to plot.")
    
    niche_df = quiche_op.adata_func[np.isin(quiche_op.adata_func.obs[annotation_key], niches)].to_df()
    niche_df[labels_key] = quiche_op.adata_func[np.isin(quiche_op.adata_func.obs[annotation_key], niches)].obs[labels_key]

    cohort_df = quiche_op.mdata['expression'].to_df()
    cohort_df[labels_key] = quiche_op.mdata['expression'].obs[labels_key]

    niche_means = niche_df.groupby(labels_key).mean().loc[:, markers]
    cohort_means = cohort_df.groupby(labels_key).mean().loc[:, markers]

    mean_diff = niche_means - cohort_means
    mean_diff.dropna(inplace=True) #remove nans if present

    adata_plot = anndata.AnnData(mean_diff)
    adata_plot.obs[labels_key] = pd.Categorical(mean_diff.index)
    adata_plot.obs_names = [f'c_{i}' for i in range(0, len(adata_plot.obs_names))]

    fig, axes = plt.subplots(1, 1, figsize = figsize, dpi = 400)

    mp = sc.pl.matrixplot(adata_plot, 
                     var_names = markers, 
                     groupby = labels_key,
                     dendrogram = dendrogram,
                     vmin = vmin,
                     vmax = vmax,
                     vcenter = vcenter,
                     cmap = cmap, 
                     colorbar_title = 'std. dev. \n (niche, cohort)', 
                     ax = axes,
                     return_fig = True)
    
    mp.add_totals(size = 2, color = 'lightgrey', show = False).style(edge_color='black', cmap = cmap)
    axes = mp.get_axes()['mainplot_ax']
    
    cbar_ax = mp.get_axes()['color_legend_ax']
    cbar_ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        if filename_save is not None:
            plt.savefig(os.path.join(save_directory, f'{filename_save}.pdf'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_directory, f'matrixplot.pdf'), bbox_inches='tight')
    else:
        plt.show()

def plot_differential_cell_type_abundance(norm_counts: pd.DataFrame,
                                        p_values_df: pd.DataFrame,
                                        condition_key: str,
                                        labels_key: str,
                                        condition_type: str = 'binary',
                                        fdr_column: str = 'FDR_p_value',
                                        threshold: float = 0.05,
                                        order: Optional[List[str]] = None,
                                        save_directory: Optional[Union[str, os.PathLike]] = None,
                                        filename_save: Optional[str] = None,
                                        n_cols: int = 7):
    """
    Plot differential abundance of cell types across conditions.
    Either uses a Wilcoxon Rank Sum test for binary covariates or Spearman rank correlation for continuous.

    Parameters
    ----------
    norm_counts: pd.DataFrame
        DataFrame containing frequencies of cell types across patient samples. Can access by calling qu.tl.differential_cell_type_abundance 
    p_values_df: pd.DataFrame
        DataFrame containing p-values, FDR-corrected p-values, and associated test statistics. Can access by calling qu.tl.differential_cell_type_abundance
    condition_key: str
        column in norm_counts containing condition-level information
    labels_key: str
        column in p_values_df containing cell phenotype information
    condition_type: str (default = 'binary')
        type of condition comparison (e.g. 'binary', 'continuous')
    fdr_column: str (default = 'FDR_p_value')
        column in p_values_df containing p-values for plotting
    threshold: float (default = 0.05)
        significance threshold for highlighting p-values
    order: list or None (default = None)
        ordering of condition labels for plotting (e.g. [0, 1])
    save_directory: Optional[Union[pathlib.Path, str]] (default = None)
        string specifying where plots should be saved. If None, will not save
    filename_save: Optional[str] (default = None)
        string specifying filename for saving. If None and save_directory is specified, will save as 'abundance'
    n_cols: int (default = 7)
        number of columns in the subplot grid

    Returns
    ----------
    None
    """    
    cell_types = p_values_df[labels_key].tolist()
    fdr_p_values = p_values_df[fdr_column].tolist()
    n_cell_types = len(cell_types)
    n_rows = (n_cell_types + n_cols - 1) // n_cols
    stat_values = p_values_df['stat_value'].tolist()
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()
    for i, (cell_type, fdr_p_value, stat_val) in enumerate(zip(cell_types, fdr_p_values, stat_values)):
        ax = axes[i]
        if condition_type == 'binary':
            sns.boxplot(data=norm_counts, x=condition_key, y=cell_type, ax=ax, width=0.5, palette="Set2", fliersize=0, order=order)
            sns.stripplot(data=norm_counts, x=condition_key, y=cell_type, ax=ax, palette="Set2", linewidth=0.2, edgecolor='gray', order=order)
            ax.set_xlabel("Outcome", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"{cell_type}", fontsize=12)
            fdr_text = f"FDR={fdr_p_value:.2g}"
            color = "red" if fdr_p_value < threshold else "black"
            
            ax.text(0.5, 0.9, f"{fdr_text}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color=color,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
        elif condition_type == 'continuous':
            sns.regplot(data=norm_counts, x=condition_key, y=cell_type, ax=ax, scatter_kws={'alpha': 0.5, 'color': 'black', 's': 20}, line_kws={'color': 'red'})
            ax.set_xlabel(f"Outcome", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"{cell_type}", fontsize=12)
            fdr_text = f"FDR={fdr_p_value:.2g}"
            stat_text  = f"ρ={stat_val:.2f}" if not pd.isna(stat_val) else "ρ=NA"
            color = "red" if fdr_p_value < threshold else "black"

            ax.text(0.5, 0.9, f"{fdr_text}\n{stat_text}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color=color,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if save_directory is not None:
        os.makedirs(save_directory, exist_ok=True)
        if filename_save is not None:
            plt.savefig(os.path.join(save_directory, f"{filename_save}.pdf"), bbox_inches='tight', dpi=400)
        else:
            plt.savefig(os.path.join(save_directory, f"abundance.pdf"), bbox_inches='tight', dpi=400)
    else:
        plt.show()