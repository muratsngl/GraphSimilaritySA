import networkx as nx
import numpy as np

import os

from skeleton_extraction import build_qap_ready_graph
from graph_modification import prune_leaves_iteratively
from skeleton import build_human36m_graph
from utils import normalize_distance_matrix
from simulated_annealing import build_affinity, simulated_annealing_restarts, SAState

# Kernel bandwidth for build_affinity (K = e^{-D/SIGMA}). With D max-normalized
# to [0, 1], SIGMA < 1 keeps the near/far affinity contrast the QAP relies on.
SIGMA = 0.2

# Weight of the repellent term in the energy function.
# Penalises IK-distant pairs landing on target-close nodes (e.g. several torso
# joints clustering on adjacent spinal bones). 0.0 = plain QAP; start at 0.2
# and raise if the solver still produces spine-clustered results.
LAMBDA_REPEL = 0.5

# Scale factor for the ancestor-ordering soft penalty.
# Each violated IK edge (where the IK child's assigned target is not a descendant
# of the IK parent's assigned target) adds gamma to the energy. Set large enough
# that one violation outweighs a typical per-pair QAP gain, but not so large that
# the SA freezes at high temperature — the annealing schedule handles the gradual
# enforcement.
GAMMA_PENALTY = 3.0


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


DEFAULT_GLB_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'assets', 'armature_01.glb')


def build_hierarchy_constraints(G_ik, ik_ordered, pruned_G, trg_ordered):
    """Build (ancestor_penalty_fn, init_fn) for soft ancestor-ordering enforcement.

    Hard filters on SA move proposals (kinematic_filter in propose_neighbor) kill
    exploration: leaf-swap and internal-swap moves almost always land on a different
    target branch, so max_tries burns out and every step becomes a no-op. The SA
    never mixes.

    Instead, ancestor violations are encoded as a SOFT ENERGY PENALTY, scaled by
    gamma_penalty. The annealing schedule then does its job: at high T the SA crosses
    violations freely to explore; at low T the penalty dominates and locks in
    structurally valid solutions.

    ancestor_penalty_fn(state) returns the number of IK edges whose ancestor
    constraint is violated in the current state. Combined with gamma_penalty in the
    SA loop: effective_energy = QAP_energy + gamma_penalty * ancestor_penalty_fn(state).

    trg_ancestors[p] = frozenset of target positions on the root→p path, excluding p.
    IK edge (parent→child) is violated when state[parent] ∉ trg_ancestors[state[child]].

    init_fn still tries to build an ancestor-consistent start by walking the IK tree
    root-first and assigning each node the shallowest available descendant of its IK
    parent's assignment. Falls back to any available node of the right class on
    topology mismatches (the penalty will guide the SA to repair violations).
    """
    ikpos = {n: i for i, n in enumerate(ik_ordered)}
    n_ik = len(ik_ordered)

    UG_ik = G_ik.to_undirected()
    ik_root = next(n for n in G_ik.nodes() if G_ik.in_degree(n) == 0)
    ik_depth = nx.shortest_path_length(UG_ik, ik_root)
    ik_bfs = sorted(G_ik.nodes(), key=lambda n: ik_depth[n])
    ik_parent = {n: next(iter(G_ik.predecessors(n)), None) for n in G_ik.nodes()}

    ik_is_leaf = {ikpos[n]: (UG_ik.degree(n) == 1) for n in G_ik.nodes()}
    ik_leaves    = np.array([p for p in range(n_ik) if     ik_is_leaf[p]], dtype=int)
    ik_internals = np.array([p for p in range(n_ik) if not ik_is_leaf[p]], dtype=int)
    ik_edges = [(ikpos[u], ikpos[v]) for u, v in G_ik.edges()]

    UG_trg = pruned_G.to_undirected()
    trg_root = next(n for n in pruned_G.nodes() if pruned_G.in_degree(n) == 0)
    n_trg = len(trg_ordered)
    trg_pos_of = {trg_ordered[p]: p for p in range(n_trg)}
    trg_is_leaf = np.array([UG_trg.degree(trg_ordered[p]) == 1 for p in range(n_trg)])

    depth_node = nx.shortest_path_length(UG_trg, trg_root)
    trg_depth  = np.array([depth_node[trg_ordered[p]] for p in range(n_trg)])

    trg_ancestors = {}
    for p in range(n_trg):
        path = nx.shortest_path(UG_trg, trg_root, trg_ordered[p])
        trg_ancestors[p] = frozenset(trg_pos_of[n] for n in path[:-1])

    def ancestor_penalty_fn(state):
        """Count IK edges whose ancestor constraint is violated."""
        violations = 0
        for parent_pos, child_pos in ik_edges:
            if state[parent_pos] not in trg_ancestors[state[child_pos]]:
                violations += 1
        return float(violations)

    def init_fn(rng):
        s = SAState(n_ik)
        s.ik_leaves    = ik_leaves
        s.ik_internals = ik_internals
        avail_leaf = [p for p in range(n_trg) if     trg_is_leaf[p]]
        avail_int  = [p for p in range(n_trg) if not trg_is_leaf[p]]
        assigned_trg = {}
        for node in ik_bfs:
            pos    = ikpos[node]
            parent = ik_parent[node]
            pool   = avail_leaf if ik_is_leaf[pos] else avail_int
            if parent is None:
                cands = list(pool)
            else:
                t_parent = assigned_trg[ikpos[parent]]
                cands = [p for p in pool if t_parent in trg_ancestors[p]]
                if not cands:
                    cands = list(pool)  # fallback: topology mismatch
            min_d = min(trg_depth[p] for p in cands)
            best  = [p for p in cands if trg_depth[p] == min_d]
            choice = int(rng.choice(best))
            s.state[pos] = choice
            assigned_trg[pos] = choice
            pool.remove(choice)
        s.unmapped_trg_leaves    = list(avail_leaf)
        s.unmapped_trg_internals = list(avail_int)
        return s

    return ancestor_penalty_fn, init_fn


def run_sa_pipeline(glb_path=DEFAULT_GLB_PATH, n_restarts=7, seed=91,
                    sigma=SIGMA, lambda_repel=LAMBDA_REPEL,
                    gamma_penalty=GAMMA_PENALTY, verbose=False):
    """Stage both skeletons and solve the correspondence QAP with SA.

    Returns a dict with everything needed to report or visualize the result:
      G_ik, ik_ordered, raw_G, pruned_G, trg_ordered,
      ik_leaves, ik_internals, best_state, best_energy, history.

    Shared by main() (console report) and visualize_mapping.py (drawing) so the
    two never drift apart on parameters.
    """
    # --- IK rig (source): skeleton.py Human3.6M skeleton -------------------- #
    G_ik, ik_ordered, D_ik = build_ik_rig()

    # --- Target (destination): pruned skeleton from the GLB ----------------- #
    raw_G, pruned_G, trg_ordered, D_trg = run_pipeline(glb_path)

    # --- Classify nodes (positional) and build affinities ------------------- #
    ik_leaves, ik_internals = classify_positions(G_ik, ik_ordered)
    trg_leaves, trg_internals = classify_positions(pruned_G, trg_ordered)

    K_ik = build_affinity(D_ik, sigma=sigma)
    K_trg = build_affinity(D_trg, sigma=sigma)

    if len(ik_leaves) > len(trg_leaves) or len(ik_internals) > len(trg_internals):
        raise ValueError(
            "Target skeleton cannot cover the IK rig per class "
            f"(need >= {len(ik_leaves)} leaves / {len(ik_internals)} internals, "
            f"have {len(trg_leaves)} / {len(trg_internals)}). "
            "Adjust pruning or relax the per-class constraint."
        )

    ancestor_penalty_fn, init_fn = build_hierarchy_constraints(
        G_ik, ik_ordered, pruned_G, trg_ordered)

    best_state, best_energy, history = simulated_annealing_restarts(
        K_ik, K_trg,
        ik_leaves, ik_internals,
        trg_leaves, trg_internals,
        n_restarts=n_restarts,
        seed=seed,
        verbose=verbose,
        kinematic_filter=None,
        init_fn=init_fn,
        penalty_fn=ancestor_penalty_fn,
        gamma_penalty=gamma_penalty,
        T=1.0, alpha=0.99, T_min=0.00001,
        iters_per_temp=100,
        lambda_repel=lambda_repel,
    )

    return {
        'G_ik': G_ik, 'ik_ordered': ik_ordered,
        'raw_G': raw_G, 'pruned_G': pruned_G, 'trg_ordered': trg_ordered,
        'ik_leaves': ik_leaves, 'ik_internals': ik_internals,
        'best_state': best_state, 'best_energy': best_energy, 'history': history,
    }


def main():
    try:
        print("\n=== STAGING + SIMULATED ANNEALING ===")
        r = run_sa_pipeline(verbose=True)

        G_ik, ik_ordered = r['G_ik'], r['ik_ordered']
        pruned_G, trg_ordered = r['pruned_G'], r['trg_ordered']
        best_state, best_energy, history = r['best_state'], r['best_energy'], r['history']

        ik_leaf_set = set(r['ik_leaves'].tolist())
        print("\n=== BEST MAPPING (IK rig -> Target) ===")
        for ik_pos, trg_pos in enumerate(best_state):
            ik_id = ik_ordered[ik_pos]
            trg_id = trg_ordered[int(trg_pos)]
            ik_name = G_ik.nodes[ik_id].get('name', str(ik_id))
            trg_name = pruned_G.nodes[trg_id].get('name', str(trg_id))
            kind = "leaf" if ik_pos in ik_leaf_set else "internal"
            print(f"  {ik_pos:>2}  {ik_name:<12s} ({kind:<8s}) -> {trg_name}")

        print("\n=== SA SUMMARY ===")
        print(f"Start energy:  {history[0]:.6f}")
        print(f"Best energy:   {best_energy:.6f}")
        print(f"Improvement:   {history[0] - best_energy:.6f}")
        print("Mapping ready for visualization.")

    except Exception as e:
        print(f"Pipeline Error: {e}")


if __name__ == "__main__":
    main()
