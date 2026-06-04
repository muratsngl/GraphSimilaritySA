import networkx as nx
from skeleton_extraction import build_qap_ready_graph
from graph_modification import prune_leaves_iteratively
from utils import normalize_distance_matrix


def run_pipeline(glb_path):
    """
    Runs the full skeleton processing pipeline.
    Returns (raw_G, pruned_G, pruned_ordered_nodes, normalized_qap_matrix).
    """
    # Phase 1: GLB Extraction
    raw_G = build_qap_ready_graph(glb_path)

    # Phase 2: Topology Modification
    pruned_G, pruned_ordered_nodes = prune_leaves_iteratively(raw_G)

    # Phase 3: Distance Matrix
    # Undirected so Floyd-Warshall can traverse edges in both directions —
    # the DiGraph only has parent->child edges, making upward/sibling paths inf.
    print("\n--- Computing Distance Matrix ---")
    undirected_G = pruned_G.to_undirected()
    pruned_distance_matrix = nx.floyd_warshall_numpy(
        undirected_G,
        nodelist=pruned_ordered_nodes,
        weight='weight'
    )

    # Phase 4: Energy Normalization
    normalized_qap_matrix = normalize_distance_matrix(pruned_distance_matrix)

    return raw_G, pruned_G, pruned_ordered_nodes, normalized_qap_matrix


def main():
    target_path = "../assets/merged-model.glb"

    try:
        raw_G, pruned_G, _, normalized_qap_matrix = run_pipeline(target_path)

        print("\n=== PIPELINE SUCCESS ===")
        print(f"Raw Graph Size:      {raw_G.number_of_nodes()} nodes")
        print(f"Pruned Graph Size:   {pruned_G.number_of_nodes()} nodes")
        print(f"Raw Max Distance:    {normalized_qap_matrix.max() * normalized_qap_matrix.max():.2f}")
        print(f"Scaled Max Distance: {normalized_qap_matrix.max():.2f}")
        print("Data is perfectly staged for Simulated Annealing.")

    except Exception as e:
        print(f"Pipeline Error: {e}")


if __name__ == "__main__":
    main()
