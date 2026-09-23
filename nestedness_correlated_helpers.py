from __future__ import annotations


import csv
import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import hashlib
import numpy as np
import importlib.abc
from collections import defaultdict
import subprocess
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt
from matplotlib import colors


# ===============

@dataclass(frozen=True)
class DiagonalSeries:
    betas: np.ndarray
    centers: np.ndarray
    center_column: str
    spread_column: str | None


@dataclass(frozen=True)
class CrossoverResult:
    exists: bool
    count: int
    first_beta: float | None
    first_left_beta: float | None
    first_right_beta: float | None
    first_kind: str | None


class _BlockNumexprFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numexpr" or fullname.startswith("numexpr."):
            raise ModuleNotFoundError("No module named 'numexpr'")
        return None

# ===============


FIELD_SEPARATOR = "\t"

_NUMEXPR_BLOCKER = _BlockNumexprFinder()

REPO_ROOT = Path(__file__).resolve().parent
JSON_SEARCH_DIRS = (
    REPO_ROOT.parent / "data", 
    REPO_ROOT / "data"
)

HYPEREDGE_OFFSET = 1_000_000

MAIN_BINARY = REPO_ROOT / "geometric-randomization" / "main"
print(f"Using GR binary at {MAIN_BINARY.resolve()}")
DIMENSION = 1

# ================


def _normalize_coordinate_values(values, dimension):
    if dimension == 1 and np.isscalar(values):
        return (float(values),)

    normalized = tuple(float(value) for value in values)
    expected_count = 1 if dimension == 1 else dimension + 1
    if len(normalized) != expected_count:
        raise ValueError(
            f"Expected {expected_count} coordinate value(s) for dimension {dimension}, "
            f"received {len(normalized)}."
        )
    return normalized



def _format_coordinate_values(values):
    return FIELD_SEPARATOR.join(f"{float(value):.17g}" for value in values)


def write_coordinate_file(
    path,
    left_nodes,
    right_nodes,
    left_coordinates,
    right_coordinates,
    *,
    dimension=1,
):
    print(f"Writing coordinate file to {path} with dimension {dimension}...")
    # print(f"First 5 left nodes and coordinates: {list(left_coordinates.items())[:5]}")
    # print(f"First 5 right nodes and coordinates: {list(right_coordinates.items())[:5]}")
    # input(f"Press Enter to continue...")

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"### D: {int(dimension)}\n")
        handle.write("### Bipartite: yes\n")
        for node in left_nodes:
            coordinate_values = _normalize_coordinate_values(
                left_coordinates[node],
                dimension,
            )
            handle.write(
                f"{node}{FIELD_SEPARATOR}A{FIELD_SEPARATOR}"
                f"{_format_coordinate_values(coordinate_values)}\n"
            )
        for node in right_nodes:
            coordinate_values = _normalize_coordinate_values(
                right_coordinates[node],
                dimension,
            )
            handle.write(
                f"{node}{FIELD_SEPARATOR}B{FIELD_SEPARATOR}"
                f"{_format_coordinate_values(coordinate_values)}\n"
            )


def sample_coordinate_map(nodes, seed, dimension=1):
    if dimension < 1:
        raise ValueError("dimension must be at least 1")

    rng = np.random.default_rng(seed)
    if dimension == 1:
        values = rng.uniform(0.0, 2.0 * math.pi, size=len(nodes))
        return {node: (float(value),) for node, value in zip(nodes, values)}

    raw_positions = rng.normal(0.0, 1.0, size=(len(nodes), dimension + 1))
    norms = np.linalg.norm(raw_positions, axis=1)
    norms[norms == 0.0] = 1.0
    normalized = raw_positions / norms[:, None]
    return {
        node: tuple(float(value) for value in row)
        for node, row in zip(nodes, normalized)
    }



def sample_correlated_hyperedge_coords(angular_pos_of_centers, nodes, 
                                       correlation_strength,
                                       seed, dimension=1):
    if dimension != 1:
        raise NotImplementedError("Only 1-dimensional correlated hyperedge coordinates implemented so far.")
    
    rng = np.random.default_rng(seed)

    if correlation_strength == 1:
        # check that positions are in [0, 2pi]
        # print(f"phi[0] is {angular_pos_of_centers[0][0]} for first center")
        assert all(0.0 <= float(phi[0]) < 2.0 * math.pi for phi in angular_pos_of_centers), "Angular positions of centers should be in [0, 2pi) for correlation_strength=1"

        center_indexes = rng.integers(0, len(angular_pos_of_centers), size=len(nodes))
        positions = [angular_pos_of_centers[idx] for idx in center_indexes]
        noise = np.zeros(len(nodes))

        print(f"Center indices (first 10): {center_indexes[:10]}")
        print(f"Positions (first 10): {positions[:10]}")
        print(f"Noise (first 10): {noise[:10]}")
        print(f"First 10 coordinate assignments: {[f'{node}: ({phi[0]:.4f},)' for node, phi in zip(nodes[:10], positions[:10])]}")
        # input(f"Press enter to continue...")

        

        return {node: (float(phi[0])+float(z), ) for (phi, z, node) in zip(positions, noise, nodes)}


    elif correlation_strength == 0: 
        positions = rng.uniform(0.0, 2.0 * math.pi, size=len(nodes))
        noise = np.zeros(len(nodes))
        return {node: (float(phi[0])+float(z), ) for (phi, z, node) in zip(positions, noise, nodes)}


    else: 
        raise NotImplementedError("Currently only supports correlation strengths of 0 or 1 ")





    


def build_shared_coordinate_file(name, left_nodes, right_nodes, 
                                 names_of_3edges=None, names_of_4edges=None,
                                 correlation_strength=0.0,
                                 output_dir=None, dimension=1):
    left_coordinates = sample_coordinate_map(
        left_nodes,
        deterministic_seed(name, "union_coords", "left", dimension),
        dimension=dimension,
    )
    # Returns: {node: (coordinate_value,)} for each node in left_nodes
    print(correlation_strength, names_of_3edges is None, names_of_4edges is None)

    if correlation_strength==0.0:
        print(f"Correlation = 0, sample as before")
        right_coordinates = sample_coordinate_map(
            right_nodes,
            deterministic_seed(name, "union_coords", "right", dimension),
            dimension=dimension,
        )

    elif correlation_strength==1 and names_of_3edges is None and names_of_4edges is None:
        right_coordinates = sample_correlated_hyperedge_coords(angular_pos_of_centers=list(
            left_coordinates.values()), 
            nodes = list(right_nodes),
            correlation_strength=correlation_strength,
            seed=deterministic_seed(name, "union_coords", "right", dimension),
            dimension=dimension)


    elif correlation_strength==1.0:
        raise ValueError("For the moment, I'm assuming this shouldn't work")
        # Check that the right nodes match the names of the union of 3-edges and 4-edges
        # assert set(right_nodes).difference(set(names_of_3edges).union(set(names_of_4edges))) == set(), "Right nodes should match the names of union of 3-edges and 4-edges"

        right_coordinates_3edges = sample_coordinate_map(
            names_of_3edges,
            deterministic_seed(name, "union_coords", "right-3", dimension),
            dimension=dimension,
        )

        right_coordinates_4edges = sample_correlated_hyperedge_coords(angular_pos_of_centers=list(
            left_coordinates.values()), 
            nodes = names_of_4edges,
            correlation_strength=correlation_strength,
            seed=deterministic_seed(name, "union_coords", "right-4", dimension),
            dimension=dimension)

        right_coordinates = {**right_coordinates_3edges, **right_coordinates_4edges}



    else:
        raise ValueError("correlation_strength should be in [0, 1] !")

    coord_path = output_dir / f"{name}-3and4.coords"
    write_coordinate_file(
        coord_path,
        left_nodes,
        right_nodes,
        left_coordinates,
        right_coordinates,
        dimension=dimension,
    )
    return coord_path



def filter_size_4_hyperedges_by_size_3_node_set(size_3_hyperedges, size_4_hyperedges):
    """Keep only size-4 edges fully contained in the size-3-layer node set."""
    size_3_nodes = {node for edge in size_3_hyperedges for node in edge}
    return [
        edge for edge in size_4_hyperedges if all(node in size_3_nodes for node in edge)
    ]


def filter_size_3_hyperedges_by_size_4_support(size_3_hyperedges, size_4_hyperedges):
    """Keep only size-3 edges fully contained in the support of the size-4 layer."""
    size_4_nodes = {node for edge in size_4_hyperedges for node in edge}
    return [
        edge for edge in size_3_hyperedges if all(node in size_4_nodes for node in edge)
    ]

def filter_size_4_hyperedges_by_size_3_support(size_3_hyperedges, size_4_hyperedges):
    """Keep only size-4 edges that intersect the support of the size-3 layer."""
    size_3_nodes = {node for edge in size_3_hyperedges for node in edge}
    return [
        edge for edge in size_4_hyperedges if any(node in size_3_nodes for node in edge)
    ]



def apply_cross_order_support_filters(
    size_3_hyperedges,
    size_4_hyperedges,
    *,
    restrict_to_common_node_core=False,
    drop_size4_without_size3_overlap=False,
    drop_size3_outside_size4_support=False,
):
    filtered_size_3 = list(size_3_hyperedges)
    filtered_size_4 = list(size_4_hyperedges)

    while True:
        changed = False

        if restrict_to_common_node_core:
            updated_size_4 = filter_size_4_hyperedges_by_size_3_node_set(
                filtered_size_3,
                filtered_size_4,
            )
            if len(updated_size_4) != len(filtered_size_4):
                filtered_size_4 = updated_size_4
                changed = True
        elif drop_size4_without_size3_overlap:
            updated_size_4 = filter_size_4_hyperedges_by_size_3_support(
                filtered_size_3,
                filtered_size_4,
            )
            if len(updated_size_4) != len(filtered_size_4):
                filtered_size_4 = updated_size_4
                changed = True

        if restrict_to_common_node_core or drop_size3_outside_size4_support:
            updated_size_3 = filter_size_3_hyperedges_by_size_4_support(
                filtered_size_3,
                filtered_size_4,
            )
            if len(updated_size_3) != len(filtered_size_3):
                filtered_size_3 = updated_size_3
                changed = True

        if not changed:
            break

    return filtered_size_3, filtered_size_4



def get_hypergraph_loader():
    if not any(isinstance(finder, _BlockNumexprFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _NUMEXPR_BLOCKER)
    sys.modules.pop("numexpr", None)

    try:
        from hypergraphx.readwrite import load_hypergraph
    except Exception as exc:
        raise RuntimeError(
            "Failed to import hypergraphx for dataset loading."
        ) from exc

    return load_hypergraph




def get_hypergraph_json_path(name):
    requested_path = Path(name)
    if requested_path.is_file():
        return requested_path.resolve()

    raw_name = requested_path.stem if requested_path.suffix == ".json" else requested_path.name
    stem_candidates = [raw_name]
    alternate_stems = (
        raw_name.replace("_", "-"),
        raw_name.replace("-", "_"),
    )
    for candidate_stem in alternate_stems:
        if candidate_stem not in stem_candidates:
            stem_candidates.append(candidate_stem)

    candidate_names = []
    for candidate_stem in stem_candidates:
        candidate_name = candidate_stem
        if not candidate_name.endswith(".json"):
            candidate_name = f"{candidate_name}.json"
        if candidate_name not in candidate_names:
            candidate_names.append(candidate_name)

    for root in JSON_SEARCH_DIRS:
        for candidate_name in candidate_names:
            input_path = root / candidate_name
            if input_path.is_file():
                return input_path

    searched_paths = ", ".join(
        str(root / candidate_name)
        for root in JSON_SEARCH_DIRS
        for candidate_name in candidate_names
    )
    raise FileNotFoundError(
        f"Missing hypergraph JSON for {name!r}. Searched: {searched_paths}"
    )




def load_hypergraph_dataset(name):
    load_hypergraph = get_hypergraph_loader()
    return load_hypergraph(str(get_hypergraph_json_path(name)))



def load_size_3_and_4_hyperedges_from_json(
    name,
    restrict_to_common_node_core=False,
    drop_size4_without_size3_overlap=False,
    drop_size3_outside_size4_support=False,
):
    hypergraph = load_hypergraph_dataset(name)
    subhypergraph = hypergraph.subhypergraph_by_orders(sizes=(3, 4), keep_nodes=False)
    size_3_hyperedges = [tuple(sorted(edge)) for edge in subhypergraph.get_edges(size=3)]
    size_4_hyperedges = [tuple(sorted(edge)) for edge in subhypergraph.get_edges(size=4)]

    if not size_3_hyperedges:
        raise ValueError(f"Hypergraph {name} has no size-3 hyperedges.")
    if not size_4_hyperedges:
        raise ValueError(f"Hypergraph {name} has no size-4 hyperedges.")

    initial_size_3_count = len(size_3_hyperedges)
    initial_size_4_count = len(size_4_hyperedges)

    if (
        restrict_to_common_node_core
        or drop_size4_without_size3_overlap
        or drop_size3_outside_size4_support
    ):
        size_3_hyperedges, size_4_hyperedges = apply_cross_order_support_filters(
            size_3_hyperedges,
            size_4_hyperedges,
            restrict_to_common_node_core=restrict_to_common_node_core,
            drop_size4_without_size3_overlap=drop_size4_without_size3_overlap,
            drop_size3_outside_size4_support=drop_size3_outside_size4_support,
        )

    if restrict_to_common_node_core:
        removed_size_3_count = initial_size_3_count - len(size_3_hyperedges)
        removed_size_4_count = initial_size_4_count - len(size_4_hyperedges)
        print(
            "Applied common-node core filter: "
            f"removed {removed_size_3_count} of {initial_size_3_count} "
            "size-3 hyperedge(s) and "
            f"{removed_size_4_count} of {initial_size_4_count} "
            "size-4 hyperedge(s) to enforce a shared order-3/order-4 node core."
        )
    elif drop_size4_without_size3_overlap:
        removed_size_4_count = initial_size_4_count - len(size_4_hyperedges)
        print(
            "Applied size-4 support-overlap filter: "
            f"removed {removed_size_4_count} of {initial_size_4_count} "
            "size-4 hyperedge(s) disjoint from the size-3 support."
        )
    if not restrict_to_common_node_core and drop_size3_outside_size4_support:
        removed_size_3_count = initial_size_3_count - len(size_3_hyperedges)
        print(
            "Applied size-3 support filter: "
            f"removed {removed_size_3_count} of {initial_size_3_count} "
            "size-3 hyperedge(s) with at least one node outside the size-4 support."
        )

    if not size_3_hyperedges:
        if restrict_to_common_node_core:
            raise ValueError(
                f"Hypergraph {name} has no size-3 hyperedges after restricting "
                "to the common order-3/order-4 node core."
            )
        if drop_size3_outside_size4_support:
            raise ValueError(
                f"Hypergraph {name} has no size-3 hyperedges after applying "
                "the size-4 support filter."
            )
        raise ValueError(f"Hypergraph {name} has no size-3 hyperedges.")
    if not size_4_hyperedges:
        if restrict_to_common_node_core:
            raise ValueError(
                f"Hypergraph {name} has no size-4 hyperedges after restricting "
                "to the common order-3/order-4 node core."
            )
        if drop_size4_without_size3_overlap:
            raise ValueError(
                f"Hypergraph {name} has no size-4 hyperedges after applying "
                "the size-3 support-overlap filter."
            )
        raise ValueError(f"Hypergraph {name} has no size-4 hyperedges.")

    return size_3_hyperedges, size_4_hyperedges



def write_bipartite_edgelist(path, edges):
    with path.open("w", encoding="utf-8") as handle:
        for left_node, right_node in edges:
            handle.write(f"{left_node}{FIELD_SEPARATOR}{right_node}\n")




def build_split_edgelists_from_hyperedges(
    name,
    output_dir,
    size_3_hyperedges,
    size_4_hyperedges,
):
    only3_edges = []
    only4_edges = []
    left_nodes = []
    seen_left_nodes = set()
    right_nodes = []
    kept_hyperedges = 0

    normalized_size_3_hyperedges = [tuple(sorted(edge)) for edge in size_3_hyperedges]
    normalized_size_4_hyperedges = [tuple(sorted(edge)) for edge in size_4_hyperedges]

    for hyperedges, target_edges in (
        (normalized_size_3_hyperedges, only3_edges),
        (normalized_size_4_hyperedges, only4_edges),
    ):
        for edge in hyperedges:
            right_node = str(HYPEREDGE_OFFSET + kept_hyperedges)
            right_nodes.append(right_node)
            kept_hyperedges += 1

            for node in edge:
                if node not in seen_left_nodes:
                    seen_left_nodes.add(node)
                    left_nodes.append(node)
                target_edges.append((node, right_node))

    if not only3_edges:
        raise ValueError(f"Hypergraph {name} has no size-3 hyperedges.")
    if not only4_edges:
        raise ValueError(f"Hypergraph {name} has no size-4 hyperedges.")


    _, only3edges_names = zip(*only3_edges)
    only3edges_names = sorted(list(set(only3edges_names)))
    _, only4edges_names = zip(*only4_edges)
    only4edges_names = sorted(list(set(only4edges_names)))

    print(len(only3edges_names), "size-3 edges before saving")
    print(len(only4edges_names), "size-4 edges before saving")
    # print(f"First 5 3-edge names: {only3edges_names[:5]}")
    # print(f"First 5 3-edge dicto items: {only3_edges[:5]}")
    # print(f"First 5 4-edge names: {only4edges_names[:5]}")
    # print(f"First 5 4-edge dicto items: {only4_edges[:5]}")

    only3_path = output_dir / f"{name}-only3.edge"
    only4_path = output_dir / f"{name}-only4.edge"

    write_bipartite_edgelist(only3_path, only3_edges)
    write_bipartite_edgelist(only4_path, only4_edges)

    return (
        only3_path,
        only4_path,
        left_nodes,
        right_nodes,
        normalized_size_3_hyperedges,
        normalized_size_4_hyperedges,
        only3edges_names,
        only4edges_names,        
    )




def build_split_edgelists_from_json(
    name,
    output_dir,
    restrict_to_common_node_core=False,
    drop_size4_without_size3_overlap=False,
    drop_size3_outside_size4_support=False,
):
    size_3_hyperedges, size_4_hyperedges = load_size_3_and_4_hyperedges_from_json(
        name,
        restrict_to_common_node_core=restrict_to_common_node_core,
        drop_size4_without_size3_overlap=drop_size4_without_size3_overlap,
        drop_size3_outside_size4_support=drop_size3_outside_size4_support,
    )
    return build_split_edgelists_from_hyperedges(
        name,
        output_dir,
        size_3_hyperedges,
        size_4_hyperedges,
    )



def compute_layer_plot_metadata(size_3_edges, size_4_edges):
    active_nodes = set()
    for edge in size_3_edges:
        active_nodes.update(edge)
    for edge in size_4_edges:
        active_nodes.update(edge)

    return {
        "node_count": int(len(active_nodes)),
        "size_3_edge_count": int(len(size_3_edges)),
        "size_4_edge_count": int(len(size_4_edges)),
        "support_upper_bound": float(
            compute_support_upper_bound(size_3_edges, size_4_edges)
        ),
    }



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



def deterministic_seed(*parts):
    encoded = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1) + 1



def ensure_main_binary():
    if not MAIN_BINARY.is_file():
        raise FileNotFoundError(f"Missing GR binary: {MAIN_BINARY}")
    if not MAIN_BINARY.stat().st_mode & 0o111:
        raise PermissionError(f"GR binary is not executable: {MAIN_BINARY}")




def detect_coordinate_dimension(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("### D:"):
                _, _, value = stripped.partition(":")
                return int(value.strip())
            if stripped.startswith("#"):
                continue
            if FIELD_SEPARATOR in stripped:
                columns = [column.strip() for column in stripped.split(FIELD_SEPARATOR)]
                if len(columns) >= 3:
                    coord_count = len(columns) - (2 if columns[1] in {"A", "B"} else 1)
                    return 1 if coord_count <= 1 else coord_count - 1
            tokens = stripped.split()
            if len(tokens) >= 2:
                coord_count = len(tokens) - (2 if len(tokens) >= 3 and tokens[1] in {"A", "B"} else 1)
                return 1 if coord_count <= 1 else coord_count - 1
            break
    raise ValueError(f"Could not determine coordinate dimension from {path}")




def run_geometric_randomization(
    input_path,
    coords_path,
    beta,
    seed,
    output_path,
    workdir,
    rewiring_multiplier=None,
    dimension=None,
):
    if dimension is None:
        dimension = detect_coordinate_dimension(coords_path)

    input_path = Path(input_path).resolve()
    coords_path = Path(coords_path).resolve()
    output_path = Path(output_path).resolve()
    workdir = Path(workdir).resolve()

    command = [
        str(MAIN_BINARY),
        "-p",
        "-d",
        str(int(dimension)),
        "-b",
        str(float(beta)),
        "-c",
        str(coords_path),
        "-s",
        str(seed),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
    ]
    if rewiring_multiplier is not None:
        command.extend(["-n", str(rewiring_multiplier)])

    try:
        result = subprocess.run(
            command,
            cwd=workdir,
            check=True,
            capture_output=False,
            text=True,
        )

    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or "No subprocess output."
        raise RuntimeError(
            f"GR failed for {input_path.name} with beta={beta}, seed={seed}. {details}"
        ) from exc

    if not output_path.is_file():
        raise RuntimeError(f"GR did not produce output edgelist: {output_path}")


def group_hyperedges_by_right_node(edges):
    grouped = defaultdict(list)
    for left_node, right_node in edges:
        grouped[right_node].append(left_node)
    return [tuple(sorted(nodes)) for nodes in grouped.values()]




def parse_bipartite_edge_columns(stripped_line, path, line_number):
    if FIELD_SEPARATOR in stripped_line:
        columns = [column.strip() for column in stripped_line.split(FIELD_SEPARATOR)]
        if len(columns) >= 2 and columns[0] and columns[1]:
            return columns[0], columns[1]
    else:
        columns = stripped_line.rsplit(None, 1)
        if len(columns) == 2 and columns[0] and columns[1]:
            return columns[0], columns[1]

    raise ValueError(
        f"Malformed bipartite edgelist line {line_number} in {path}: "
        f"could not parse two node labels from {stripped_line!r}"
    )



def read_bipartite_edgelist(path):
    edges = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            left_node, right_node = parse_bipartite_edge_columns(
                stripped,
                path,
                line_number,
            )
            edges.append((left_node, right_node))
    return edges

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


def save_original_nestedness_txt(path, original_nestedness):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{float(original_nestedness):.10f}\n")


def save_run_config_json(
    path,
    *,
    analysis,
    name,
    n_realisations,
    workers_requested,
    workers_used,
    rewiring_multiplier,
    dimension=None,
    extra_config=None,
):
    config = {
        "analysis": analysis,
        "dataset": name,
        "dimension": int(DIMENSION if dimension is None else dimension),
        "num_realisations": int(n_realisations),
        "workers_requested": workers_requested,
        "workers_used": int(workers_used),
        "rewiring_multiplier": rewiring_multiplier,
    }
    if extra_config:
        config.update(extra_config)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def split_hyperedges_by_size(hyperedges):
    size_3_edges = [edge for edge in hyperedges if len(edge) == 3]
    size_4_edges = [edge for edge in hyperedges if len(edge) == 4]

    invalid_sizes = sorted({len(edge) for edge in hyperedges if len(edge) not in {3, 4}})
    if invalid_sizes:
        raise ValueError(f"Unexpected hyperedge sizes after merge: {invalid_sizes}")

    return size_3_edges, size_4_edges


def compute_support_upper_bound(size_3_edges, size_4_edges):
    """Compute U_support^(3,4), the support-based upper bound on nestedness."""
    if not size_3_edges:
        return float("nan")

    nodes_4 = set().union(*size_4_edges) if size_4_edges else set()
    feasible_count = sum(1 for edge in size_3_edges if all(node in nodes_4 for node in edge))
    return float(feasible_count / len(size_3_edges))



def compute_size_3_containment_multiplicity(size_3_edges, size_4_edges):
    containment_index = {}
    for idx, edge in enumerate(size_4_edges):
        for node in edge:
            containment_index.setdefault(node, set()).add(idx)

    multiplicities = []
    for edge in size_3_edges:
        candidate_sets = [containment_index.get(node, set()) for node in edge]
        if any(len(candidate_set) == 0 for candidate_set in candidate_sets):
            multiplicities.append(0)
            continue
        multiplicities.append(len(set.intersection(*candidate_sets)))
    return np.asarray(multiplicities, dtype=int)


def compute_subset_overlap_diagnostics(size_3_edges, size_4_edges):
    induced_size_3_subsets = set()
    for edge in size_4_edges:
        for subset in combinations(edge, 3):
            induced_size_3_subsets.add(tuple(sorted(subset)))

    if not induced_size_3_subsets:
        return {
            "induced_size_3_subset_count": 0,
            "realized_induced_size_3_subset_count": 0,
            "subset_overlap_ratio": float("nan"),
        }

    realized_size_3_edges = {tuple(sorted(edge)) for edge in size_3_edges}
    realized_overlap = realized_size_3_edges & induced_size_3_subsets
    return {
        "induced_size_3_subset_count": int(len(induced_size_3_subsets)),
        "realized_induced_size_3_subset_count": int(len(realized_overlap)),
        "subset_overlap_ratio": float(len(realized_overlap) / len(induced_size_3_subsets)),
    }


def compute_component_diagnostics(size_3_edges, size_4_edges):
    adjacency = {}

    def connect(left_node, right_node):
        adjacency.setdefault(left_node, set()).add(right_node)
        adjacency.setdefault(right_node, set()).add(left_node)

    for edge_idx, edge in enumerate(size_3_edges):
        hyperedge_node = f"E3::{edge_idx}"
        for node in edge:
            connect(f"L::{node}", hyperedge_node)

    for edge_idx, edge in enumerate(size_4_edges):
        hyperedge_node = f"E4::{edge_idx}"
        for node in edge:
            connect(f"L::{node}", hyperedge_node)

    if not adjacency:
        return {
            "component_count": 0,
            "giant_component_edge_share": float("nan"),
        }

    remaining = set(adjacency)
    component_edge_totals = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        degree_sum = 0
        while stack:
            current = stack.pop()
            neighbors = adjacency[current]
            degree_sum += len(neighbors)
            unseen = neighbors & remaining
            remaining -= unseen
            stack.extend(unseen)
        component_edge_totals.append(degree_sum // 2)

    total_edges = sum(component_edge_totals)
    giant_edge_share = float("nan")
    if total_edges > 0:
        giant_edge_share = float(max(component_edge_totals) / total_edges)
    return {
        "component_count": int(len(component_edge_totals)),
        "giant_component_edge_share": giant_edge_share,
    }




def compute_structural_diagnostics(size_3_edges, size_4_edges):
    size_3_count = len(size_3_edges)
    size_4_count = len(size_4_edges)

    nodes_3 = set().union(*size_3_edges) if size_3_edges else set()
    nodes_4 = set().union(*size_4_edges) if size_4_edges else set()
    union_nodes = nodes_3 | nodes_4

    node_jaccard = float("nan")
    if union_nodes:
        node_jaccard = float(len(nodes_3 & nodes_4) / len(union_nodes))

    # This is U_support^(3,4): a necessary-condition upper bound on nestedness.
    feasible_fraction = compute_support_upper_bound(size_3_edges, size_4_edges)

    degree_3 = Counter()
    degree_4 = Counter()
    for edge in size_3_edges:
        degree_3.update(edge)
    for edge in size_4_edges:
        degree_4.update(edge)

    degree_correlation = float("nan")
    if union_nodes:
        xs = np.asarray([degree_3[node] for node in sorted(union_nodes)], dtype=float)
        ys = np.asarray([degree_4[node] for node in sorted(union_nodes)], dtype=float)
        if np.std(xs) > 0.0 and np.std(ys) > 0.0:
            degree_correlation = float(np.corrcoef(xs, ys)[0, 1])

    multiplicities = compute_size_3_containment_multiplicity(size_3_edges, size_4_edges)
    mean_containers = float("nan")
    multiplicity_share_0 = float("nan")
    multiplicity_share_1 = float("nan")
    multiplicity_share_2 = float("nan")
    multiplicity_share_3_plus = float("nan")
    if multiplicities.size > 0:
        mean_containers = float(np.mean(multiplicities))
        multiplicity_share_0 = float(np.mean(multiplicities == 0))
        multiplicity_share_1 = float(np.mean(multiplicities == 1))
        multiplicity_share_2 = float(np.mean(multiplicities == 2))
        multiplicity_share_3_plus = float(np.mean(multiplicities >= 3))

    diagnostics = {
        "size_3_edge_count": int(size_3_count),
        "size_4_edge_count": int(size_4_count),
        "fraction_size_3_supported_by_order_4_nodes": feasible_fraction,
        "order_3_4_node_jaccard": node_jaccard,
        "order_3_4_degree_correlation": degree_correlation,
        "mean_containers_per_size3": mean_containers,
        "multiplicity_share_0": multiplicity_share_0,
        "multiplicity_share_1": multiplicity_share_1,
        "multiplicity_share_2": multiplicity_share_2,
        "multiplicity_share_3_plus": multiplicity_share_3_plus,
    }
    diagnostics.update(compute_subset_overlap_diagnostics(size_3_edges, size_4_edges))
    diagnostics.update(compute_component_diagnostics(size_3_edges, size_4_edges))
    return diagnostics




def compute_diagonal_diagnostics(
    dataset_name,
    betas,
    centers,
    original_nestedness,
    size_3_edges,
    size_4_edges,
    extra_metadata=None,
):
    betas = np.asarray(betas, dtype=float)
    centers = np.asarray(centers, dtype=float)

    crossover = summarize_crossover(betas, centers, original_nestedness)
    max_index = int(np.argmax(centers))
    min_index = int(np.argmin(centers))

    diagnostics = {
        "dataset": dataset_name,
        "original_nestedness": float(original_nestedness),
        "diagonal_min_nestedness": float(centers[min_index]),
        "diagonal_min_beta": float(betas[min_index]),
        "diagonal_max_nestedness": float(centers[max_index]),
        "diagonal_max_beta": float(betas[max_index]),
        "diagonal_crossover_exists": bool(crossover.exists),
        "diagonal_crossover_count": int(crossover.count),
        "first_crossover_beta": _nan_if_undefined(crossover.first_beta),
        "first_crossover_beta_left": _nan_if_undefined(crossover.first_left_beta),
        "first_crossover_beta_right": _nan_if_undefined(crossover.first_right_beta),
        "first_crossover_kind": crossover.first_kind or "",
    }
    # diagnostics.update(compute_structural_diagnostics(size_3_edges, size_4_edges))
    if extra_metadata:
        diagnostics.update(extra_metadata)
    return diagnostics


def _nan_if_undefined(value):
    if value is None:
        return float("nan")
    return float(value)



def summarize_crossover(betas, centers, target, atol=1e-12):
    betas = np.asarray(betas, dtype=float)
    centers = np.asarray(centers, dtype=float)
    crossings = []

    for beta, value in zip(betas, centers):
        if math.isclose(float(value), float(target), rel_tol=0.0, abs_tol=atol):
            crossings.append(
                {
                    "beta": float(beta),
                    "beta_left": float(beta),
                    "beta_right": float(beta),
                    "kind": "exact",
                }
            )

    for idx in range(len(betas) - 1):
        left_value = float(centers[idx] - target)
        right_value = float(centers[idx + 1] - target)

        if math.isclose(left_value, 0.0, rel_tol=0.0, abs_tol=atol):
            continue
        if math.isclose(right_value, 0.0, rel_tol=0.0, abs_tol=atol):
            continue
        if left_value * right_value > 0:
            continue

        left_beta = float(betas[idx])
        right_beta = float(betas[idx + 1])
        crossing_fraction = abs(left_value) / (abs(left_value) + abs(right_value))
        crossing_beta = left_beta + crossing_fraction * (right_beta - left_beta)
        crossings.append(
            {
                "beta": float(crossing_beta),
                "beta_left": left_beta,
                "beta_right": right_beta,
                "kind": "interpolated",
            }
        )

    if not crossings:
        return CrossoverResult(
            exists=False,
            count=0,
            first_beta=None,
            first_left_beta=None,
            first_right_beta=None,
            first_kind=None,
        )

    first = min(crossings, key=lambda item: item["beta"])
    return CrossoverResult(
        exists=True,
        count=len(crossings),
        first_beta=float(first["beta"]),
        first_left_beta=float(first["beta_left"]),
        first_right_beta=float(first["beta_right"]),
        first_kind=str(first["kind"]),
    )





def save_diagnostics_csv(path, diagnostics):
    fieldnames = [
        "dataset",
        "original_nestedness",
        "diagonal_min_nestedness",
        "diagonal_min_beta",
        "diagonal_max_nestedness",
        "diagonal_max_beta",
        "diagonal_crossover_exists",
        "diagonal_crossover_count",
        "first_crossover_beta",
        "first_crossover_beta_left",
        "first_crossover_beta_right",
        "first_crossover_kind",
        "size_3_edge_count",
        "size_4_edge_count",
        "fraction_size_3_supported_by_order_4_nodes",
        "order_3_4_node_jaccard",
        "order_3_4_degree_correlation",
        "component_count",
        "giant_component_edge_share",
        "induced_size_3_subset_count",
        "realized_induced_size_3_subset_count",
        "subset_overlap_ratio",
        "mean_containers_per_size3",
        "multiplicity_share_0",
        "multiplicity_share_1",
        "multiplicity_share_2",
        "multiplicity_share_3_plus",
        "dimension",
        "coordinate_source",
        "filter_variant",
    ]
    extra_fields = sorted(set(diagnostics) - set(fieldnames))
    fieldnames.extend(extra_fields)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: diagnostics.get(field, "") for field in fieldnames})



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


def plot_1_panel(dataArr, 
                 xticklabels, yticklabels,
                 xlabel, ylabel,
                 title,
                 ax=None,
                  log_option = False,
                  vmin=None, vmax=None,
                 step=1,
                 cmap='BuGn'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.get_figure()
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




def HO_convert_node_labels_to_integers(H:hgx.Hypergraph)->hgx.Hypergraph:
    
    def relabel(edges: list, relabeling: dict):
        """
        Relabel the vertices of a hypergraph according to a given relabeling

        Parameters
        ----------
        edges : list
            Edges of the hypergraph
        relabeling : dict
            Relabeling

        Returns
        -------
        list
            Edges of the hypergraph with the vertices relabeled

        Notes
        -----
        The relabeling is a dictionary that maps the old labels to the new labels
        """
        res = []
        for edge in edges:
            new_edge = []
            for v in edge:
                new_edge.append(relabeling[v])
            res.append(tuple(sorted(new_edge)))
        return sorted(res)



    oldelist = H.get_edges()

    mapping = {old:new for (old,new) in zip(H.get_nodes(), range(H.num_nodes()))}
    
    newelist = relabel(edges=oldelist, relabeling=mapping)

    
    Hnew = hgx.Hypergraph(newelist)
    return Hnew



