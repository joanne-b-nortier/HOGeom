import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import colors
# import cmrameri.cm as cm
import matplotlib as mp
import pandas as pd
import os


def get_crossover_data(ds, DATAROOT):


    indices = [el for el in os.listdir(DATAROOT / ds / f"correlation_strength_0.00") if 'ations' not in el] 

    DIFFS = []
    vals_for_std = []

    for idx in indices:
        generated_nestedness = pd.read_csv(DATAROOT / ds / f"correlation_strength_0.00/{idx}/{ds}_nestedness_diagonal_mean_std.csv")[['beta', 'mean_nestedness']]
        original_nestedness = np.loadtxt(DATAROOT / ds / f"correlation_strength_0.00/{idx}/{ds}_original_nestedness.txt")
        diff_nestedness = generated_nestedness.copy()
        diff_nestedness['abs_diff'] = np.abs(generated_nestedness['mean_nestedness'] - original_nestedness)
        approx_crossover_beta = diff_nestedness.iloc[np.argmin(diff_nestedness['abs_diff'])].beta
        diff_nestedness['diff'] = (generated_nestedness['mean_nestedness'] - original_nestedness)
        vals_for_std.append(generated_nestedness['mean_nestedness'].values)

        if len(DIFFS) == 0:
            DIFFS = diff_nestedness.copy()
        else:
            DIFFS = DIFFS + diff_nestedness.copy()
        
    spreads = np.std(np.array(vals_for_std), axis=0)
    assert len(spreads) == len(diff_nestedness['beta']), "careful. You took the standard deviation of the wrong axis"

    DIFFS = DIFFS / len(indices)
    DIFFS['std_nestedness'] = spreads
    DIFFS

    return DIFFS


def plot_diff_overall_with_crossover(DATAROOT, ds, ax=None):
    
    generated_nestedness = pd.read_csv(DATAROOT/f"{ds}_nestedness_diagonal_mean_std.csv")[['beta', 'mean_nestedness']]
    spreads = pd.read_csv(DATAROOT / f"{ds}_nestedness_diagonal_mean_std.csv")[['beta', 'std_nestedness']]
    original_nestedness = np.loadtxt(DATAROOT / f"{ds}_original_nestedness.txt")
    diagonal_crossover_exists = pd.read_csv(DATAROOT / f"{ds}_nestedness_diagonal_diagnostics.csv", header=0, index_col=0).diagonal_crossover_exists.values[0]
    DIFFS = generated_nestedness.copy()
    DIFFS['abs_diff'] = np.abs(generated_nestedness['mean_nestedness'] - original_nestedness)
    if diagonal_crossover_exists:
        approx_crossover_beta = DIFFS.iloc[np.argmin(DIFFS['abs_diff'])].beta
    else:
        approx_crossover_beta = None
    DIFFS['diff'] = (generated_nestedness['mean_nestedness'] - original_nestedness)
    DIFFS['std_nestedness'] = spreads['std_nestedness']
    
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    else:
        fig = ax.get_figure()
    ax.plot(DIFFS['beta'], DIFFS['diff'], 
            marker='o', 
            markersize=5,
            color="navy", 
            linewidth = 2.5)
    ax.fill_between(DIFFS['beta'], DIFFS['diff'] - DIFFS['std_nestedness'], DIFFS['diff'] + DIFFS['std_nestedness'],
                    color='navy', alpha=0.2)
    ax.set_ylabel(rf"${{I}}_{{\mathcal{{H}}}}^{{GR}} - I_{{\mathcal{{H}}}}^{{{{emp}}}}$")
    ax.set_xlabel(r"$\beta$")
    y_bottom, y_top = ax.get_ylim()
    
    if approx_crossover_beta is not None:
        ax.axvline(approx_crossover_beta,
                #    ymin=y_bottom, ymax=y_top,
                color="k",
                    linestyle=":",
                    # label=rf"$\beta^{{*}} \approx{approx_crossover_beta:.3f}$", 
                    linewidth=4.0)
        ax.text(.9, .1, rf"$\beta^{{*}} \approx{approx_crossover_beta:.3f}$",
                ha='right', va='bottom', 
                transform=ax.transAxes, 
                fontsize=20, color='r')
        ax.fill_betweenx(y=[y_bottom, y_top], x1=approx_crossover_beta, x2=ax.get_xlim()[1], alpha=0.25, color="mediumpurple")
    ax.axhline(0, 
               xmin=0, xmax=max(DIFFS['beta']),
               color='red', linestyle='--', 
               linewidth=6.0, 
               zorder=0)
    

    
    
    # ax.legend(frameon=False, loc = 'lower right', fontsize=20, labelcolor='r')
    ax.set_xlim(0, max(DIFFS['beta']))
    ax.set_ylim(y_bottom, y_top)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel(ax.get_xlabel(), fontsize=26)
    ax.set_ylabel(ax.get_ylabel(), fontsize=26)

    return fig, ax, approx_crossover_beta, DIFFS



def bipartite_to_hypergraph(bipartite_edges, metadata=False):
    Warning("This function assumes no metadata, so duplicate hyperedges are ignored.")
    hyperedges_dict = defaultdict(list)

    for node, hyperedge_id in bipartite_edges:
        hyperedges_dict[hyperedge_id].append(node)

    # Convert grouped nodes back to hyperedge tuples
    hypergraph_edges = [tuple(nodes) for nodes in hyperedges_dict.values()]
    
    if metadata is True:
        return hgx.Hypergraph(edge_list=hypergraph_edges, edge_metadata=list(hyperedges_dict.keys()))
    else:
        return hgx.Hypergraph(hypergraph_edges)





def plot_matching_diagonal(title, ax, rootdir, ds):
    rootdir = str(rootdir)
    ax.set_title(title)
    diagnostics = pd.read_csv(f"{rootdir}/{ds}_nestedness_diagonal_diagnostics.csv", header=0, index_col=0)
    original_nestedness = diagnostics['original_nestedness'].values[0]
    crossover_beta = diagnostics['first_crossover_beta'].values[0]
    
    fname = f"{rootdir}/{ds}_nestedness_diagonal_mean_std.csv"
    if not os.path.isfile(fname):
        fname = f"{rootdir}/{ds}_nestedness_diagonal_median_iqr.csv"
        if not os.path.isfile(fname):
            print(f"Failure for {ds}: no mean/std nor median/iqr file exists")
            return ax
    
    df = pd.read_csv(fname, header=0)

    x = df['beta'].values
    if 'median' in fname:
        y = df['median_nestedness'].values
        ystd = df['iqr_nestedness'].values
    elif 'mean' in fname:
        y = df['mean_nestedness'].values
        ystd = df['std_nestedness'].values


    ax.plot(x, y, 
            c='navy', 
            marker='.', 
            label='Mean randomised nestedness' if 'mean' in fname else 'Mean randomised nestedness'
            )
    ax.fill_between(x, y-ystd, y+ystd, color="navy",
                alpha=0.15,
                linewidth=0)

    ax.axhline(
            original_nestedness,
            color="red",
            linestyle="--",
            linewidth=2.0,
            label=rf"$\mathcal{{I}}^{{(3,4)}}$ = {original_nestedness:.4f}",
        )

    if not np.isnan(crossover_beta):
        ax.axvline(
                    crossover_beta,
                    color="forestgreen",
                    linestyle=":",
                    linewidth=2.0,
                    label=f"Beta crossover ≈ {crossover_beta:.3f}",
                )
        if crossover_beta < 2.5:
            ax.set_xlim(0, 5)
        elif crossover_beta < 5: 
            ax.set_xlim(0, 10)
    else:
        ax.axhline(
                    max(y),
                    color="darkorange",
                    linestyle=":",
                    linewidth=2.0,
                    label=f"Maximum I = {max(y):.4f}",
                )
    ax.grid(True, alpha=0.25)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='best')

    return ax


def plot_zscore(ax, fig, zscore_I, maxlevel, 
                cmap, 
                cmap_onesided='Blues'):
    """
    mostly a function for the custom colormap, but also does the plotting of the zscore matrix
    """
    
    zscore_I[np.isinf(abs(zscore_I))] = np.nan
    zscore_I = zscore_I[:maxlevel, :maxlevel].T

    # print(f"ds={ds}, (nanmin, nanmax for zscore plot: ({np.nanmin(zscore_I)}, {np.nanmax(zscore_I)})")
    
    if np.nanmin(zscore_I)<0:
        norm = colors.SymLogNorm(
                vmin=np.nanmin(zscore_I), 
                vmax=np.nanmax(zscore_I), 
                linthresh=0.0001, # Values in (-0.0001, 0.0001) are linear
                base=10)
        cmap2 = cmap
        cmap2.set_bad(color='lightgray', alpha=1.)

        # now set the 0's to be -inf so that they'll be white in the plot
        zscore_I[zscore_I == 0] = np.nanmin(zscore_I) - 100
        cmap2.set_under('w')

        im = ax.imshow(zscore_I, cmap=cmap2, origin='lower', norm=norm)
        cb = fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)

        ticks = cb.get_ticks()
        cb.set_ticks(ticks[1::2])


        
    else: 
        # print(np.nanmin(zscore_I), np.nanmax(zscore_I))
        norm = colors.SymLogNorm(vmin=0.0001, 
                            vmax=np.nanmax(zscore_I), 
                            linthresh=0.0001, 
                            base=10)
        cmap2 = mp.colormaps.get_cmap(cmap_onesided)
        cmap2.set_bad(color='lightgray', alpha=1.)

        # now set the 0's to be -inf so that they'll be white in the plot
        zscore_I[zscore_I == 0] = -1
        cmap2.set_under('w')

        im = ax.imshow(zscore_I, cmap=cmap2, origin='lower', norm=norm)
        cb = fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)


    ax.set_xlabel("Z-score")

    # remove lower and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # add ticks only to y axis
    # ax.set_xticks(range(0, maxlevel+1, 5 ))
    # ax.set_xticklabels(np.array(range(0, maxlevel+1, 5 ))+2)
    
    # ax.set_yticks(range(0, maxlevel+1, 5 ))
    # ax.set_yticklabels(np.array(range(0, maxlevel+1, 5 ))+2)


def plot_1_panel(dataArr, 
                 xticklabels=None, yticklabels=None,
                 xlabel='m', ylabel='n',
                 title='',
                 ax=None,
                  log_option = True,
                  vmin=None, vmax=None,
                 step=1,
                 cmap=plt.get_cmap('BuGn')):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.get_figure()

    if xticklabels is None:
        xticklabels = list(range(2, dataArr.shape[1]+2))
    if yticklabels is None:
        yticklabels = list(range(2, dataArr.shape[0]+2))

    if log_option:
        if vmin is None:
            vmin = np.nanmin(dataArr[dataArr>0])
        if vmax is None:
            vmax = np.nanmax(dataArr[dataArr>0])

    cmap.set_bad(color='lightgray')
    im = ax.imshow(dataArr, 
                      norm= colors.LogNorm(vmin=vmin, vmax=vmax) if log_option else None, 
                      cmap=cmap, origin='lower', aspect='auto')
    plt.colorbar(im, fraction=0.036)
    ax.set_yticks(ticks=np.arange(stop=len(yticklabels), step=step), labels=(yticklabels[::step]))
    ax.set_xticks(ticks=np.arange(stop=len(xticklabels), step=step), labels=xticklabels[::step], rotation=90)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # title = r'$\log I_{ln}$' if log_option else r'$\langle I_{ln} \rangle$'  
    ax.set_title(title, fontsize=18);
    return fig, ax


def reshape_array(arr: np.ndarray, desired_shape: tuple, pad_value=0) -> np.ndarray:
    """
    Pads or cuts a numpy array to match the desired shape.
    Works for any number of dimensions.
    
    Args:
        arr:           Input array of any shape
        desired_shape: Target shape tuple
        pad_value:     Value to use for padding (default 0)
    
    Returns:
        Array with exactly the desired shape
    """
    assert arr.ndim == len(desired_shape), (
        f"Number of dimensions must match: got {arr.ndim}, expected {len(desired_shape)}"
    )

    # ── Step 1: cut down any dims that are too large ──────────────────────────
    slices = tuple(slice(0, s) for s in desired_shape)
    result = arr[slices]

    # ── Step 2: pad any dims that are too small ───────────────────────────────
    pad_width = [(0, max(0, desired_shape[i] - result.shape[i]))
                 for i in range(len(desired_shape))]
    result = np.pad(result, pad_width, mode="constant", constant_values=pad_value)

    return result


def compute_nestedness(size_small_edges, size_large_edges):
    if not size_small_edges or not size_large_edges:
        return 0.0

    containment_index = defaultdict(set)
    for idx, edge in enumerate(size_large_edges):
        for node in edge:
            containment_index[node].add(idx)

    contained_size_3_edges = 0
    for edge in size_small_edges:
        candidate_sets = [containment_index[node] for node in edge if node in containment_index]
        if len(candidate_sets) != len(edge):
            continue
        intsctn = set.intersection(*candidate_sets)
        # if we're going to be adding a set that's a subset of itself, ignore
        intsctn = {idx for idx in intsctn if size_large_edges[idx] != edge}
        if intsctn:
            contained_size_3_edges += 1

    return float(contained_size_3_edges / len(size_small_edges))


import hypergraphx as hgx
def calc_fingerprint_nestedness(H: hgx.Hypergraph,
                                max_level=30) -> np.ndarray:
    """Calculate the fingerprint nestedness of a hypergraph H."""

    # previously had not accounted for empty rows ... sighs
    max_level = min(max_level, H.max_size()) if max_level is not None else H.max_size()
    sizes = list(range(2, max_level+1))

    I_matrix = np.empty((len(sizes), len(sizes)))
    I_matrix[:] = np.nan
    for i, size_m in enumerate(sizes):
        for j, size_n in (enumerate(sizes)):
            if size_m < size_n:
                _I = compute_nestedness(size_small_edges= H.get_edges(size=size_m), 
                                                    size_large_edges= H.get_edges(size=size_n))
                I_matrix[i, j] = _I


    return I_matrix

