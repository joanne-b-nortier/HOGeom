import argparse
import concurrent.futures
import csv
import json
import os
import tempfile
from pathlib import Path
import hypergraphx as hgx
import cmcrameri.cm as cm

_MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "geometric-randomization-mpl-cache"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from nestedness_correlated_helpers import (
    build_shared_coordinate_file,
    compute_diagonal_diagnostics,
    load_hypergraph_dataset,
    compute_layer_plot_metadata,
    compute_nestedness,
    deterministic_seed,
    ensure_main_binary,
    group_hyperedges_by_right_node,
    read_bipartite_edgelist,
    run_geometric_randomization,
    save_original_nestedness_txt,
    save_run_config_json,
    reshape_array,
    HO_convert_node_labels_to_integers,
    calc_fingerprint_nestedness,
    save_diagnostics_csv,
)

BETAS_3 = np.linspace(0.0, 10.0, num=50)
N_REALISATIONS = 10

REPO_ROOT = Path(__file__).resolve()
RESULTS_ROOT = REPO_ROOT / "output" / "whole-network"

SUMMARY_MODES = {
    "mean_std": {
        "csv_center": "mean_nestedness",
        "csv_spread": "std_nestedness",
        "center_label": "mean",
        "band_label": "Mean +/- std. dev.",
        "line_label": "Mean randomized nestedness",
        "title_label": "mean +/- std",
    },
    "median_iqr": {
        "csv_center": "median_nestedness",
        "csv_spread": "iqr_nestedness",
        "center_label": "median",
        "band_label": "Interquartile range",
        "line_label": "Median randomized nestedness",
        "title_label": "median + IQR",
    },
}



def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate a nestedness diagonal sweep for one hypergraph dataset, "
            "assuming beta_3 = beta_4."
        )
    )
    parser.add_argument(
        "--name", 
        required=True, 
        help="Dataset name without suffix.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum number of beta workers. Defaults to CPU count.",
    )
    parser.add_argument(
        "--min-beta",
        type=float,
        default=float(BETAS_3[0]),
        help=f"Minimum beta value. Default: {float(BETAS_3[0]):.6f}",
    )
    parser.add_argument(
        "--max-beta",
        type=float,
        default=float(BETAS_3[-1]),
        help=f"Maximum beta value. Default: {float(BETAS_3[-1]):.6f}",
    )
    parser.add_argument(
        "--num-betas",
        type=int,
        default=len(BETAS_3),
        help=f"Number of beta values along the diagonal. Default: {len(BETAS_3)}",
    )
    parser.add_argument(
        "--realisations",
        type=int,
        default=N_REALISATIONS,
        help=f"Number of GR realisations per beta value. Default: {N_REALISATIONS}",
    )
    parser.add_argument(
        "-n",
        "--rewiring-multiplier",
        type=positive_int_arg,
        default=None,
        help=(
            "Override the GR rewiring budget multiplier passed to `main -n`. "
            "If omitted, the GR binary default is used."
        ),
    )
    parser.add_argument(
        "--summary",
        type=normalize_summary_mode,
        default="mean_std",
        help=(
            "Summary statistic for the randomized nestedness values. "
            "Accepted values: mean_std (alias: mean+std) or "
            "median_iqr (alias: median+iqr). Default: mean_std"
        ),
    )
    parser.add_argument(
        "--correlation-strength",
        type=float,
        default=1.0,
        help=(
            "Correlation strength between size-3 and size-4 hyperedge angular "
            "positions, in [0, 1]. When 0, becomes independent uniform sampling, "
            "when 1, perfectly correlated coordinates where each hyperedge coordinates"
            "are copied from a randomly chosen center coordinate. Default: 0.0"
        ),
    )
    parser.add_argument(
        "--restrict-LCC", 
        action="store_true",
        help=(
            "Restrict the size-3 and size-4 layers to their largest connected "
            "component before building the diagonal sweep. "
        ),
    )
    parser.add_argument(
        "--max-level", 
        type=int,
        default=30,
        help=(
            "Maximum hyperedge size for the nestedness fingerprint I calculation. "
        )
    )
    parser.add_argument(
        '--save-idx', 
        type=int, 
        default=None, 
        help=("Assuming I want to save multiple runs and later average, we'll have have to change the folder we save to. " \
        "This index will be appended to the folder name to distinguish different runs. Default: 0 (no change / overwrite).")
    )
    parser.add_argument(
        '--RESULTS_ROOT',
        type=str,
        default=str(RESULTS_ROOT),
        help="Root directory for saving results. Default: 'results/whole-network' in the repository.",
    )
    
    return parser.parse_args(argv)




def normalize_summary_mode(value):
    normalized = value.strip().lower().replace("+", "_").replace("-", "_")
    if normalized not in SUMMARY_MODES:
        valid = ", ".join(["mean_std", "mean+std", "median_iqr", "median+iqr"])
        raise argparse.ArgumentTypeError(
            f"Unsupported summary mode '{value}'. Use one of: {valid}."
        )
    return normalized


def positive_int_arg(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def summarize_nestedness(values, summary_mode):
    values = np.asarray(values, dtype=float)
    if summary_mode == "mean_std":
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=0))
        return mean_value, std_value, mean_value - std_value, mean_value + std_value

    q25, median_value, q75 = np.percentile(values, [25, 50, 75])
    return float(median_value), float(q75 - q25), float(q25), float(q75)


def build_beta_values(min_beta, max_beta, num_betas):
    if num_betas < 1:
        raise ValueError("--num-betas must be at least 1.")
    if max_beta < min_beta:
        raise ValueError("--max-beta must be greater than or equal to --min-beta.")
    return np.linspace(min_beta, max_beta, num=num_betas)


def process_beta_value(
    name,
    correlation_strength,
    beta_idx,
    beta,
    n_realisations,
    summary_mode,
    rewiring_multiplier,
    input_path_hyperedges,
    shared_coords_path,
    temp_root,
    I_shape,
    max_level,
    RESULTS_ROOT
):
    beta_dir = temp_root / f"beta_{beta_idx:04d}"
    beta_dir.mkdir(parents=True, exist_ok=True)

    I_matrices = np.zeros((n_realisations, *I_shape), dtype=float)

    realisation_values = []
    for realisation_idx in range(n_realisations):
        output_all = beta_dir / f"randomized_r_{realisation_idx}_ALL.edge"
        # output_only4 = beta_dir / f"randomized_r_{realisation_idx}_only4.edge"

        run_geometric_randomization(
            input_path=input_path_hyperedges,
            coords_path=shared_coords_path,
            beta=beta,
            seed=deterministic_seed(name, "diag", "gr", "all", beta_idx, realisation_idx),
            output_path=output_all,
            workdir=beta_dir,
            rewiring_multiplier=rewiring_multiplier,
        )
        # run_geometric_randomization(
        #     input_path=input_only4,
        #     coords_path=shared_coords_path,
        #     beta=beta,
        #     seed=deterministic_seed(name, "diag", "gr", 4, beta_idx, realisation_idx),
        #     output_path=output_only4,
        #     workdir=beta_dir,
        #     rewiring_multiplier=rewiring_multiplier,
        # )

        merged_edges = read_bipartite_edgelist(output_all)
        # merged_edges.extend(read_bipartite_edgelist(output_only4))

        hyperedges = group_hyperedges_by_right_node(merged_edges)
        # size_3_edges, size_4_edges = split_hyperedges_by_size(hyperedges)
        realisation_values.append(compute_nestedness(hyperedges, hyperedges))

        _H = hgx.Hypergraph(list(hyperedges))
        _H = _H.subhypergraph_by_orders(sizes=list(range(2, max_level+1)))
        I_matrices[realisation_idx] = reshape_array(calc_fingerprint_nestedness(H=_H, max_level=max_level), I_shape)

    
    # save one of the realisations for later inspection if beta is larger than 15
    # if beta > 15:
    output_dir = RESULTS_ROOT /  "realisations"      # f"correlation_strength_{correlation_strength:.2f}" /
    output_dir.mkdir(parents=True, exist_ok=True)
    output_all_final = output_dir / f"randomized_beta_{beta:.3f}_r_{realisation_idx}_ALL.edge"
    with output_all_final.open("w", encoding="utf-8") as handle:
        for left_node, right_node in merged_edges:
            handle.write(f"{left_node} {right_node}\n")

    center_value, spread_value, lower_band, upper_band = summarize_nestedness(
        realisation_values,
        summary_mode,
    )


    center_I_value = np.mean(I_matrices, axis=0)
    spread_I_value = np.std(I_matrices, axis=0, ddof=0)
    lower_band_I = center_I_value - spread_I_value
    upper_band_I = center_I_value + spread_I_value



    return (
        beta_idx,
        center_value,
        spread_value,
        lower_band,
        upper_band,
        center_I_value,
        spread_I_value,
        lower_band_I,
        upper_band_I,
        np.asarray(realisation_values, dtype=float),
    )


def save_diagonal_csv(path, betas, centers, spreads, summary_mode, realisation_matrix=None):
    summary_meta = SUMMARY_MODES[summary_mode]
    header = ["beta", summary_meta["csv_center"], summary_meta["csv_spread"]]

    if realisation_matrix is not None:
        realisation_matrix = np.asarray(realisation_matrix, dtype=float)
        if realisation_matrix.ndim != 2:
            raise ValueError("realisation_matrix must be a 2D array when provided.")
        if realisation_matrix.shape[0] != len(betas):
            raise ValueError(
                "realisation_matrix row count must match the number of beta values."
            )
        header.extend(
            [f"realisation_{idx:03d}" for idx in range(realisation_matrix.shape[1])]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_idx, (beta, center_value, spread_value) in enumerate(zip(betas, centers, spreads)):
            row = [f"{beta:.6f}", f"{center_value:.10f}", f"{spread_value:.10f}"]
            if realisation_matrix is not None:
                row.extend(f"{value:.10f}" for value in realisation_matrix[row_idx])
            writer.writerow(row)


def save_diagonal_plot(
    path,
    betas,
    centers,
    lower_bands,
    upper_bands,
    name,
    original_nestedness,
    summary_mode,
    diagnostics,
    layer_metadata,
):
    summary_meta = SUMMARY_MODES[summary_mode]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        betas,
        centers,
        color="navy",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label=summary_meta["line_label"],
    )
    if np.any(upper_bands > lower_bands):
        ax.fill_between(
            betas,
            lower_bands,
            upper_bands,
            color="navy",
            alpha=0.15,
            linewidth=0,
            label=summary_meta["band_label"],
        )
    ax.axhline(
        original_nestedness,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"Ground truth = {original_nestedness:.4f}",
    )
    if diagnostics["diagonal_crossover_exists"]:
        crossover_beta = diagnostics["first_crossover_beta"]
        ax.axvline(
            crossover_beta,
            color="forestgreen",
            linestyle=":",
            linewidth=2.0,
            label=f"First crossover beta ≈ {crossover_beta:.3f}",
        )
    ax.set_xlabel(r"$\beta$ with $\beta_3 = \beta_4$")
    ax.set_ylabel("Nestedness")
    crossover_status = "crossover found" if diagnostics["diagonal_crossover_exists"] else "no crossover"
    ax.set_title(
        f"{name} nestedness diagonal sweep ({summary_meta['title_label']}, {crossover_status})\n"
        f"original = {original_nestedness:.4f}, "
        # f"U_support^(3,4) = {layer_metadata['support_upper_bound']:.4f}\n"
        f"|V| = {layer_metadata['node_count']}, "
        f"|E| = {layer_metadata['edge_count']}, "
        # f"|E4| = {layer_metadata['size_4_edge_count']}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_output_paths(output_dir, name, summary_mode):
    base_name = f"{name}_nestedness_diagonal"
    generic_csv_path = output_dir / f"{base_name}.csv"
    generic_plot_path = output_dir / f"{base_name}.png"
    summary_csv_path = output_dir / f"{base_name}_{summary_mode}.csv"
    summary_plot_path = output_dir / f"{base_name}_{summary_mode}.png"
    return generic_csv_path, generic_plot_path, summary_csv_path, summary_plot_path


def build_diagonal_sweep(
    name,
    correlation_strength, 
    workers=None,
    betas=None,
    n_realisations=N_REALISATIONS,
    summary_mode="median_iqr",
    rewiring_multiplier=None,
    max_level=None,
    restrict_to_LCC=False,
    save_idx=None,
    RESULTS_ROOT=None
    # restrict_to_common_node_core=False,
    # drop_size4_without_size3_overlap=False,
    # drop_size3_outside_size4_support=False,
):
    ensure_main_binary()

    if betas is None:
        betas = np.asarray(BETAS_3, dtype=float)
    else:
        betas = np.asarray(betas, dtype=float)

    centers = np.zeros(len(betas), dtype=float)
    mean_I_matrices_per_beta = {}
    spreads = np.zeros(len(betas), dtype=float)
    lower_bands = np.zeros(len(betas), dtype=float)
    upper_bands = np.zeros(len(betas), dtype=float)
    centers_I = {}
    spreads_I = {}

    realisation_matrix = np.zeros((len(betas), n_realisations), dtype=float)
    summary_meta = SUMMARY_MODES[summary_mode]
    total_betas = len(betas)
    max_workers = workers or (os.cpu_count() or 1)
    max_workers = max(1, min(max_workers, total_betas))

    with tempfile.TemporaryDirectory(prefix=f"{name}_nestedness_diagonal_") as temp_dir:
        temp_root = Path(temp_dir)
        # (
        #     input_only3,
        #     input_only4,
        #     left_nodes,
        #     right_nodes,
        #     original_size_3_hyperedges,
        #     original_size_4_hyperedges,
        #     only3edges_names,
        #     only4edges_names,        
        # ) = build_split_edgelists_from_json(
        #     name,
        #     temp_root,
        #     restrict_to_common_node_core=restrict_to_common_node_core,
        #     drop_size4_without_size3_overlap=drop_size4_without_size3_overlap,
        #     drop_size3_outside_size4_support=drop_size3_outside_size4_support,
        # )
        hypergraph = load_hypergraph_dataset(name)
        edges_to_remove = [edge for edge in hypergraph.get_edges() if len(edge) > max_level]
        hypergraph.remove_edges(edges_to_remove)
        assert hypergraph.max_size() <= max_level, f"All hyperedges must have size at most {max_level} after filtering."
        if restrict_to_LCC:
            print(f"RESTRICTED TO LCC:\npreviously {hypergraph.num_nodes()} nodes and {hypergraph.num_edges()} edges.")
            hypergraph = hypergraph.subhypergraph_largest_component()
            print(f"now {hypergraph.num_nodes()} nodes and {hypergraph.num_edges()} edges.")
        hypergraph = HO_convert_node_labels_to_integers(hypergraph)
        
        left_nodes = hypergraph.get_nodes()
        offset = max(left_nodes) + 1
        right_nodes = [el + offset for el in range(hypergraph.num_edges())]

        original_nestedness_overall = compute_nestedness(
            list(hypergraph.get_edges()),
            list(hypergraph.get_edges()),
        )
        original_nestedness_I = calc_fingerprint_nestedness(H=hypergraph, max_level=max_level)
        layer_metadata = {
            "node_count": int(len(left_nodes)),
            "edge_count": int(len(right_nodes)),
        }
       
        shared_coords_path = build_shared_coordinate_file(name=name, 
                                                          left_nodes=left_nodes, 
                                                          right_nodes=right_nodes, 
                                                          names_of_3edges=None, 
                                                          names_of_4edges=None,
                                                          correlation_strength=correlation_strength,
                                                          output_dir=temp_root)
        print(f"Original nestedness all: {original_nestedness_overall:.6f}")
        print(
            f"Running {total_betas} beta value(s) with {max_workers} worker(s) "
            f"and {n_realisations} realisation(s) per beta "
            f"using {summary_mode}. "
            f"Node count={layer_metadata['node_count']}, Edge count={layer_metadata['edge_count']}. "
            # f"Common-node core filter: "
            # f"{'enabled' if restrict_to_common_node_core else 'disabled'}. "
            # f"Size-4 overlap filter: "
            # f"{'enabled' if drop_size4_without_size3_overlap else 'disabled'}. "
            # f"Size-3 support filter: "
            # f"{'enabled' if drop_size3_outside_size4_support else 'disabled'}. "
            f"GR rewiring multiplier: "
            f"{rewiring_multiplier if rewiring_multiplier is not None else 'default'}."
        )

        input_path_edges = temp_root / f"{name}-hyperedges.edge"
        # write bip network to feed into GR, with right node IDs as edges and left nodes IDs as nodes
        with input_path_edges.open("w", encoding="utf-8") as handle:
            for i, (right_node, hyperedge) in enumerate(zip(right_nodes, hypergraph.get_edges())):
                for left_node in hyperedge:
                    handle.write(f"{left_node} {right_node}\n")

        if not os.path.isdir(RESULTS_ROOT):
            (RESULTS_ROOT).mkdir(parents=True, exist_ok=True)
        with (RESULTS_ROOT / f"{name}-hyperedges.edge").open("w", encoding="utf-8") as handle:
            for i, (right_node, hyperedge) in enumerate(zip(right_nodes, hypergraph.get_edges())):
                for left_node in hyperedge:
                    handle.write(f"{left_node} {right_node}\n")

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for beta_idx, beta in enumerate(betas):
                futures.append(
                    executor.submit(
                        process_beta_value,
                        name,
                        correlation_strength,
                        beta_idx,
                        float(beta),
                        n_realisations,
                        summary_mode,
                        rewiring_multiplier,
                        input_path_edges,
                        shared_coords_path,
                        temp_root,
                        original_nestedness_I.shape, 
                        max_level, 
                        RESULTS_ROOT
                    )
                )

            with tqdm(
                total=total_betas,
                desc="Diagonal sweep",
                unit="beta",
                dynamic_ncols=True,
            ) as progress:
                for future in concurrent.futures.as_completed(futures):
                    (
                        beta_idx,
                        center_value,
                        spread_value,
                        lower_band,
                        upper_band,
                        center_I_value,
                        spread_I_value,
                        lower_band_I,
                        upper_band_I,
                        realisation_values,
                    ) = future.result()
                    centers[beta_idx] = center_value
                    spreads[beta_idx] = spread_value
                    lower_bands[beta_idx] = lower_band
                    upper_bands[beta_idx] = upper_band
                    centers_I[beta] = center_I_value
                    spreads_I[beta] = spread_I_value
                    realisation_matrix[beta_idx] = realisation_values
                    progress.set_postfix(
                        {
                            "last_beta": f"{betas[beta_idx]:.3f}",
                            summary_meta["center_label"]: f"{center_value:.6f}",
                        }
                    )
                    progress.update(1)

    if save_idx is None:
        output_dir = RESULTS_ROOT # / f"correlation_strength_{correlation_strength:.2f}" / f"LCC_{int(restrict_to_LCC)}" 
    else:
        output_dir = RESULTS_ROOT / f"idx_{save_idx}"    # / f"correlation_strength_{correlation_strength:.2f}" / f"LCC_{int(restrict_to_LCC)}
    output_dir.mkdir(parents=True, exist_ok=True)

    generic_csv_path, generic_plot_path, summary_csv_path, summary_plot_path = (
        build_output_paths(output_dir, name, summary_mode)
    )
    config_path = output_dir / f"{name}_nestedness_diagonal_config.json"
    original_nestedness_path = output_dir / f"{name}_original_nestedness.txt"
    original_nestedness_path_I = output_dir / f"{name}_original_nestedness_I.txt"
    diagnostics_path = output_dir / f"{name}_nestedness_diagonal_diagnostics.csv"
    diagnostics = compute_diagonal_diagnostics(
        name,
        betas,
        centers,
        original_nestedness_overall,
        None, #original_size_3_hyperedges,
        None, #original_size_4_hyperedges,
    )
    # save_diagonal_csv(
    #     generic_csv_path,
    #     betas,
    #     centers,
    #     spreads,
    #     summary_mode,
    #     realisation_matrix=realisation_matrix,
    # )
    save_diagonal_csv(
        summary_csv_path,
        betas,
        centers,
        spreads,
        summary_mode,
        realisation_matrix=realisation_matrix,
    )
    # save_diagonal_plot(
    #     generic_plot_path,
    #     betas,
    #     centers,
    #     lower_bands,
    #     upper_bands,
    #     name,
    #     original_nestedness_all,
    #     summary_mode,
    #     diagnostics,
    #     layer_metadata,
    # )
    save_diagonal_plot(
        summary_plot_path,
        betas,
        centers,
        lower_bands,
        upper_bands,
        name,
        original_nestedness_overall,
        summary_mode,
        diagnostics,
        layer_metadata,
    )
    save_run_config_json(
        config_path,
        analysis="nestedness_diagonal",
        name=name,
        n_realisations=n_realisations,
        workers_requested=workers,
        workers_used=max_workers,
        rewiring_multiplier=rewiring_multiplier,
        extra_config={
            "beta3_equals_beta4": True,
            "beta_max": float(betas[-1]),
            "beta_min": float(betas[0]),
            "beta_values": [float(beta) for beta in betas],
            # "restrict_to_common_node_core": bool(restrict_to_common_node_core),
            # "drop_size4_without_size3_overlap": bool(
                # drop_size4_without_size3_overlap
            # ),
            # "drop_size3_outside_size4_support": bool(
                # drop_size3_outside_size4_support
            # ),
            "node_count": layer_metadata["node_count"],
            "num_betas": int(len(betas)),
            "edge_count": layer_metadata["edge_count"],
            # "size_4_edge_count": layer_metadata["size_4_edge_count"],
            # "support_upper_bound_3_4": layer_metadata["support_upper_bound"],
            "summary_mode": summary_mode,
            "correlation_strength": correlation_strength,
        },
    )
    save_original_nestedness_txt(original_nestedness_path, original_nestedness_overall)
    np.savetxt(original_nestedness_path_I, original_nestedness_I)
    save_diagnostics_csv(diagnostics_path, diagnostics)

    np.save(output_dir / f"{name}_diagonal_centers_I.npy", centers_I)
    np.save(output_dir / f"{name}_diagonal_spreads_I.npy", spreads_I)
    with open(output_dir / f"{name}_overall_nestednesses_per_beta.txt", "w") as f:
        json.dump({beta: nestedness for beta, nestedness in zip(betas, realisation_matrix.mean(axis=1))}, f)


    if diagnostics["diagonal_crossover_exists"]:
        print(
            "Diagonal crossover found "
            f"near beta={diagnostics['first_crossover_beta']:.6f} "
            f"({diagnostics['first_crossover_kind']})."
        )
    else:
        print("No diagonal crossover found within the sampled beta range.")
        
    

    print(f"Saved diagonal sweep to {generic_csv_path}")
    print(f"Saved diagonal sweep to {summary_csv_path}")
    print(f"Saved diagonal plot to {generic_plot_path}")
    print(f"Saved diagonal plot to {summary_plot_path}")
    print(f"Saved diagonal config to {config_path}")
    print(f"Saved original nestedness to {original_nestedness_path}")
    print(f"Saved diagonal diagnostics to {diagnostics_path}")
    
    last_beta = betas[-1]
    plot_I_panel_for_ds(sampled_I_mean=centers_I[last_beta], 
                        true_I=original_nestedness_I, 
                        maxlevel=max_level, 
                        ds=f"{name} at beta={last_beta}", 
                        save_fname=output_dir / f"{name}_I_panel.png")

    return betas, centers, spreads


from nestedness_correlated_helpers import plot_1_panel
def plot_I_panel_for_ds(sampled_I_mean, true_I, maxlevel=30, ds="", save_fname=None):
    startpos = 0
    matrix = np.nan_to_num(true_I[startpos:, startpos:].T) + np.nan_to_num(sampled_I_mean[startpos:, startpos:])
    fig, ax = plot_1_panel(dataArr=matrix[:maxlevel, :maxlevel], 
                xticklabels=range(startpos+2, len(matrix)+2), 
                yticklabels=range(startpos+2, len(matrix)+2), cmap=cm.devon_r, 
                xlabel="HOCM", ylabel="True", 
                vmin=np.nanmin(matrix[np.nonzero(matrix)]), vmax=np.nanmax(matrix[np.nonzero(matrix)]),
                log_option=True,
                title=rf"{ds}")

    fig.savefig(save_fname, dpi=200)


def main(argv=None):
    args = parse_args(argv)
    betas = build_beta_values(args.min_beta, args.max_beta, args.num_betas)
    build_diagonal_sweep(
        name=args.name,
        correlation_strength=args.correlation_strength,
        workers=args.workers,
        betas=betas,
        n_realisations=args.realisations,
        summary_mode=args.summary,
        rewiring_multiplier=args.rewiring_multiplier,
        restrict_to_LCC=args.restrict_LCC,
        max_level=args.max_level,
        save_idx=args.save_idx,
        RESULTS_ROOT=Path(args.RESULTS_ROOT),
        # restrict_to_common_node_core=args.restrict_to_common_node_core,
        # drop_size4_without_size3_overlap=args.drop_size4_without_size3_overlap,
        # drop_size3_outside_size4_support=args.drop_size3_outside_size4_support,
    )


if __name__ == "__main__":
    main()
