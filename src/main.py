import networkx as nx
import numpy as np

import os

from skeleton_extraction import build_qap_ready_graph
from graph_modification import prune_leaves_iteratively
from skeleton import build_human36m_graph
from utils import normalize_distance_matrix
from simulated_annealing import build_affinity, simulated_annealing


# --------------------------------------------------------------------------- #
# Shared graph -> QAP helpers
#
# Both the SA state array and the affinity matrices are indexed by POSITION in
# an ordered node list, not by raw node id (target ids are sparse after
# pruning). Everything below therefore works in positional space and only maps
# back to ids/names when reporting the final assignment.
# --------------------------------------------------------------------------- #

def graph_to_normalized_matrix(G, ordered_nodes):
    """Undirected geodesic distance matrix over `ordered_nodes`, max-normalized.

    Undirected so Floyd-Warshall can traverse parent<->child both ways; the
    DiGraph only stores parent->child edges, which would leave upward/sibling
    paths at infinity.
    """
    undirected_G = G.to_undirected()
    D = nx.floyd_warshall_numpy(undirected_G, nodelist=ordered_nodes, weight='weight')
    return normalize_distance_matrix(D)


def classify_positions(G, ordered_nodes):
    """Split node POSITIONS (indices into `ordered_nodes`) into leaves/internals
    by undirected degree: degree 1 == leaf (end-effector), degree > 1 == internal.

    Returns (leaf_positions, internal_positions) as int numpy arrays, ready to
    feed straight into simulated_annealing().
    """
    UG = G.to_undirected()
    leaves, internals = [], []
    for pos, node in enumerate(ordered_nodes):
        if UG.degree(node) == 1:
            leaves.append(pos)
        else:
            internals.append(pos)
    return np.array(leaves, dtype=int), np.array(internals, dtype=int)


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #

def run_pipeline(glb_path):
    """Full target-skeleton pipeline from a GLB file.

    Returns (raw_G, pruned_G, pruned_ordered_nodes, normalized_qap_matrix).
    """
    # Phase 1: GLB Extraction
    raw_G = build_qap_ready_graph(glb_path)

    # Phase 2: Topology Modification
    pruned_G, pruned_ordered_nodes = prune_leaves_iteratively(raw_G)

    # Phase 3 + 4: Distance Matrix + Energy Normalization
    print("\n--- Computing Distance Matrix ---")
    normalized_qap_matrix = graph_to_normalized_matrix(pruned_G, pruned_ordered_nodes)

    return raw_G, pruned_G, pruned_ordered_nodes, normalized_qap_matrix


def build_ik_rig():
    """The IK reference rig: the Human3.6M 17-joint skeleton from skeleton.py.

    Returns (G_ik, ik_ordered_nodes, D_ik) where D_ik is the normalized
    geodesic distance matrix over ik_ordered_nodes. Not pruned — it is already
    the minimal reference configuration.
    """
    G_ik = build_human36m_graph()
    ik_ordered_nodes = sorted(G_ik.nodes())
    D_ik = graph_to_normalized_matrix(G_ik, ik_ordered_nodes)
    return G_ik, ik_ordered_nodes, D_ik


def main():
    target_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'merged-model.glb')

    try:
        # --- IK rig (source): skeleton.py Human3.6M skeleton ---------------- #
        G_ik, ik_ordered, D_ik = build_ik_rig()

        # --- Target (destination): pruned skeleton from the GLB ------------- #
        raw_G, pruned_G, trg_ordered, D_trg = run_pipeline(target_path)

        # --- Classify nodes (positional) and build affinities --------------- #
        ik_leaves, ik_internals = classify_positions(G_ik, ik_ordered)
        trg_leaves, trg_internals = classify_positions(pruned_G, trg_ordered)

        K_ik = build_affinity(D_ik)
        K_trg = build_affinity(D_trg)

        print("\n=== STAGING ===")
        print(f"IK rig:  {len(ik_ordered)} nodes "
              f"({len(ik_leaves)} leaves, {len(ik_internals)} internals)")
        print(f"Target:  {len(trg_ordered)} nodes "
              f"({len(trg_leaves)} leaves, {len(trg_internals)} internals)")

        if len(ik_leaves) > len(trg_leaves) or len(ik_internals) > len(trg_internals):
            raise ValueError(
                "Target skeleton cannot cover the IK rig per class "
                f"(need >= {len(ik_leaves)} leaves / {len(ik_internals)} internals, "
                f"have {len(trg_leaves)} / {len(trg_internals)}). "
                "Adjust pruning or relax the per-class constraint."
            )

        # --- Run Simulated Annealing ---------------------------------------- #
        print("\n--- Running Simulated Annealing ---")
        best_state, best_energy, history = simulated_annealing(
            K_ik, K_trg,
            ik_leaves, ik_internals,
            trg_leaves, trg_internals,
            T=1.0, alpha=0.99, T_min=0.00001,
            iters_per_temp=100,
            seed=71,
            verbose=False,
        )

        # --- Report the mapping (positions -> ids -> names) ----------------- #
        ik_leaf_set = set(ik_leaves.tolist())
        print("\n=== BEST MAPPING (IK rig -> Target) ===")
        for ik_pos, trg_pos in enumerate(best_state):
            ik_id = ik_ordered[ik_pos]
            trg_id = trg_ordered[int(trg_pos)]
            ik_name = G_ik.nodes[ik_id].get('name', str(ik_id))
            trg_name = pruned_G.nodes[trg_id].get('name', str(trg_id))
            kind = "leaf" if ik_pos in ik_leaf_set else "internal"
            print(f"  {ik_name:<12s} ({kind:<8s}) -> {trg_name}")

        print("\n=== SA SUMMARY ===")
        print(f"Start energy:  {history[0]:.6f}")
        print(f"Best energy:   {best_energy:.6f}")
        print(f"Improvement:   {history[0] - best_energy:.6f}")
        print("Mapping ready for visualization.")

    except Exception as e:
        print(f"Pipeline Error: {e}")


if __name__ == "__main__":
    main()
