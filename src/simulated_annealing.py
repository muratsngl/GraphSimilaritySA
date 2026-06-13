"""
Simulated Annealing for Skeleton Correspondence QAP.

Maps every node of the (smaller) IK skeleton to a distinct node of the
(larger-or-equal) Target skeleton, while strictly preserving leaf/internal
topology and honoring an injective mapping with an "unmapped pool" for the
extra Target nodes.

Pipeline (matches the design spec):
  Phase 1  Pre-computation & classification (affinity matrices + node classes)
  Phase 2  State representation & injective initialization
  Phase 3  Objective / energy function
  Phase 4  Tri-state transposition engine (leaf swap / internal swap / injection)
  Phase 5  The annealing loop (thermostat)

Energy uses the negated QAP maximization objective plus an optional mismatch
penalty term so the loop minimizes:

    E(Pi) = -sum_{i,j} K_ik[i,j] * K_trg[Pi(i),Pi(j)]
            + lambda * sum_{i,j} |K_ik[i,j] - K_trg[Pi(i),Pi(j)]|

The mismatch term penalises any asymmetry between IK affinity and target
affinity — both far-IK-on-close-target (clustering) and close-IK-on-far-target
(chain skipping) incur the same cost. lambda_repel=0.0 recovers the plain QAP.
"""

import numpy as np


# --------------------------------------------------------------------------- #
# Phase 1: Pre-Computation & Classification
# --------------------------------------------------------------------------- #

def build_affinity(normalized_distance_matrix, sigma=0.1):
    """K = e^{-D/sigma}. Rewards local matches, penalizes distant noise.

    Expects a normalized [0, 1] distance matrix (see utils.normalize_distance_matrix).

    `sigma` is the kernel bandwidth. With D already max-normalized to [0, 1],
    sigma=1.0 squashes K into [e^-1, 1] ~= [0.37, 1], washing out the near/far
    contrast the QAP objective relies on. Smaller sigma restores it: sigma=0.1
    gives D/sigma in [0, 10] -> K in [~4.5e-5, 1], the full dynamic range that
    Holzschuh gets for free from raw (un-normalized) geodesics.
    """
    D = np.asarray(normalized_distance_matrix, dtype=np.float64)
    return np.exp(-D / sigma)


def classify_nodes(adjacency_or_degrees):
    """Split node indices into leaves (degree 1) and internals (degree > 1).

    Accepts either:
      - a 1D array/list of integer degrees indexed by node id, or
      - a 2D adjacency / affinity-style matrix (degree inferred from nonzero
        off-diagonal entries).

    Returns (leaves, internals) as 1D int numpy arrays of node ids.
    """
    arr = np.asarray(adjacency_or_degrees)

    if arr.ndim == 1:
        degrees = arr.astype(int)
    elif arr.ndim == 2:
        mask = arr != 0
        np.fill_diagonal(mask, False)
        degrees = mask.sum(axis=1).astype(int)
    else:
        raise ValueError("Expected 1D degree array or 2D adjacency matrix.")

    node_ids = np.arange(degrees.shape[0])
    leaves = node_ids[degrees == 1]
    internals = node_ids[degrees > 1]
    return leaves.astype(int), internals.astype(int)


# --------------------------------------------------------------------------- #
# Phase 2: State Representation & Injective Initialization
# --------------------------------------------------------------------------- #

class SAState:
    """Holds the active mapping plus the unmapped Target pools.

    state[i] == target id assigned to IK node i.
    Leaves map only to Target leaves; internals only to Target internals.
    """

    __slots__ = (
        "state",
        "ik_leaves", "ik_internals",
        "unmapped_trg_leaves", "unmapped_trg_internals",
    )

    def __init__(self, n_ik):
        self.state = np.full(n_ik, -1, dtype=int)
        self.ik_leaves = None
        self.ik_internals = None
        self.unmapped_trg_leaves = []
        self.unmapped_trg_internals = []

    def copy(self):
        new = SAState(self.state.shape[0])
        new.state = self.state.copy()
        new.ik_leaves = self.ik_leaves
        new.ik_internals = self.ik_internals
        new.unmapped_trg_leaves = list(self.unmapped_trg_leaves)
        new.unmapped_trg_internals = list(self.unmapped_trg_internals)
        return new


def initialize_state(ik_leaves, ik_internals,
                     trg_leaves, trg_internals,
                     rng):
    """Build a valid T=0 starting state obeying topological constraints.

    Shuffle each Target class, assign the first len(ik_class) entries into the
    state at the IK class indices, and stash the remainder in the unmapped pool.
    """
    ik_leaves = np.asarray(ik_leaves, dtype=int)
    ik_internals = np.asarray(ik_internals, dtype=int)
    trg_leaves = np.asarray(trg_leaves, dtype=int)
    trg_internals = np.asarray(trg_internals, dtype=int)

    if len(ik_leaves) > len(trg_leaves):
        raise ValueError(
            f"Not enough Target leaves ({len(trg_leaves)}) "
            f"to cover IK leaves ({len(ik_leaves)})."
        )
    if len(ik_internals) > len(trg_internals):
        raise ValueError(
            f"Not enough Target internals ({len(trg_internals)}) "
            f"to cover IK internals ({len(ik_internals)})."
        )

    n_ik = len(ik_leaves) + len(ik_internals)
    s = SAState(n_ik)
    s.ik_leaves = ik_leaves
    s.ik_internals = ik_internals

    # Leaves
    shuffled_leaves = trg_leaves.copy()
    rng.shuffle(shuffled_leaves)
    s.state[ik_leaves] = shuffled_leaves[:len(ik_leaves)]
    s.unmapped_trg_leaves = list(shuffled_leaves[len(ik_leaves):])

    # Internals
    shuffled_internals = trg_internals.copy()
    rng.shuffle(shuffled_internals)
    s.state[ik_internals] = shuffled_internals[:len(ik_internals)]
    s.unmapped_trg_internals = list(shuffled_internals[len(ik_internals):])

    return s


# --------------------------------------------------------------------------- #
# Phase 3: The Objective Function (Energy)
# --------------------------------------------------------------------------- #

def energy(state_array, K_ik, K_trg, lambda_repel=0.0,
           lat_ik=None, lat_trg=None, lambda_lateral=0.0):
    """E = -sum K_ik * K_trg
          + lambda       * sum |K_ik - K_trg_reordered|
          - lambda_lat   * sum_i lat_ik[i] . lat_trg[state[i]]

    Attraction term: rewards IK-close pairs mapping to target-close pairs.
    Mismatch term: penalises any asymmetry between IK affinity and target
    affinity — both far-IK-on-close-target (spine clustering) and
    close-IK-on-far-target (chain skipping) cost the same lambda per pair.
    Lateral term: rewards assignments where the IK joint and its assigned
    target joint point in the same 2D direction from their respective roots
    (cosine similarity in XY plane). Breaks left/right symmetry degeneracy
    that the QAP alone cannot resolve. lat_ik and lat_trg are (n, 2) arrays
    of unit vectors; joints at the origin have zero vectors and contribute 0.
    lambda_repel=0.0 and lambda_lateral=0.0 recover the original QAP exactly.
    """
    reordered_K_trg = K_trg[np.ix_(state_array, state_array)]
    attraction = np.sum(K_ik * reordered_K_trg)
    e = -attraction
    if lambda_repel != 0.0:
        mismatch = np.sum(np.abs(K_ik - reordered_K_trg))
        e += lambda_repel * mismatch
    if lambda_lateral != 0.0 and lat_ik is not None and lat_trg is not None:
        lateral_alignment = np.sum(lat_ik * lat_trg[state_array])
        e -= lambda_lateral * lateral_alignment
    return e


# --------------------------------------------------------------------------- #
# Phase 4: The Tri-State Transposition Engine (Swapper)
# --------------------------------------------------------------------------- #

def propose_neighbor(current, rng, kinematic_filter=None, max_tries=32):
    """Return a new SAState produced by exactly one constrained move.

      Move A  Leaf swap        (swap two active IK-leaf assignments)
      Move B  Internal swap    (swap two active IK-internal assignments)
      Move C  Subgraph injection (swap an active Target node for an unmapped one)

    Move choice is random but skips moves that are impossible for the current
    state (e.g. injection with an empty pool). `kinematic_filter`, if given, is
    called as filter(new_state_array) -> bool; a False result rejects the
    proposal and another move is attempted. This is the hook for a future
    "Kinematic Limb-Crossing Filter".
    """
    for _ in range(max_tries):
        candidate = _single_move(current, rng)
        if candidate is None:
            continue
        if kinematic_filter is not None and not kinematic_filter(candidate.state):
            continue
        return candidate
    # Fell through: return an unchanged copy so the loop can keep cooling.
    return current.copy()


def _single_move(current, rng):
    """Execute one randomly chosen move; return a new SAState or None."""
    moves = []
    if len(current.ik_leaves) >= 2:
        moves.append("leaf_swap")
    if len(current.ik_internals) >= 2:
        moves.append("internal_swap")
    if current.unmapped_trg_leaves or current.unmapped_trg_internals:
        moves.append("injection")

    if not moves:
        return None

    move = rng.choice(moves)
    new = current.copy()

    if move == "leaf_swap":
        idx1, idx2 = rng.choice(current.ik_leaves, size=2, replace=False)
        new.state[idx1], new.state[idx2] = new.state[idx2], new.state[idx1]

    elif move == "internal_swap":
        idx1, idx2 = rng.choice(current.ik_internals, size=2, replace=False)
        new.state[idx1], new.state[idx2] = new.state[idx2], new.state[idx1]

    else:  # injection
        # Pick a class that actually has an unmapped node available.
        classes = []
        if current.unmapped_trg_leaves and len(current.ik_leaves) > 0:
            classes.append("leaf")
        if current.unmapped_trg_internals and len(current.ik_internals) > 0:
            classes.append("internal")
        if not classes:
            return None

        if rng.choice(classes) == "leaf":
            ik_idx = int(rng.choice(current.ik_leaves))
            pool = new.unmapped_trg_leaves
        else:
            ik_idx = int(rng.choice(current.ik_internals))
            pool = new.unmapped_trg_internals

        pop_pos = rng.integers(len(pool))
        new_trg_val = pool.pop(pop_pos)
        old_trg_val = new.state[ik_idx]
        new.state[ik_idx] = new_trg_val
        pool.append(int(old_trg_val))

    return new


# --------------------------------------------------------------------------- #
# Phase 5: The Simulated Annealing Loop (Thermostat)
# --------------------------------------------------------------------------- #

def simulated_annealing(K_ik, K_trg,
                        ik_leaves, ik_internals,
                        trg_leaves, trg_internals,
                        T=1.0, alpha=0.999, T_min=0.0000001,
                        iters_per_temp=1,
                        lambda_repel=0.0,
                        lat_ik=None, lat_trg=None, lambda_lateral=0.0,
                        penalty_fn=None,
                        gamma_penalty=0.0,
                        kinematic_filter=None,
                        init_fn=None,
                        seed=None,
                        verbose=False):
    """Run SA and return (best_state_array, best_energy, history).

    Parameters
    ----------
    K_ik, K_trg : affinity matrices from build_affinity().
    ik_leaves, ik_internals, trg_leaves, trg_internals : node-id arrays.
    T, alpha, T_min : initial temperature, geometric cooling rate, stop threshold.
    iters_per_temp : SA steps taken at each temperature before cooling.
    lambda_repel : weight of the repellent term (see energy()). 0.0 = plain QAP.
    penalty_fn : optional callable(state_array) -> float returning a structural
        violation count or score. Added to the energy as gamma_penalty * penalty_fn(state).
        Encodes soft constraints (e.g. ancestor ordering) that should guide but not
        hard-block exploration. At high T the SA crosses violations freely; at low T
        the penalty dominates and locks in valid solutions.
    gamma_penalty : scale factor for penalty_fn. Set large enough that one violation
        outweighs a typical per-pair QAP gain.
    kinematic_filter : optional callable(state_array) -> bool for hard move rejection.
        Reserved for constraints that must never be violated (e.g. type class).
        Do NOT use for ancestor/depth ordering — that kills exploration.
    init_fn : optional callable(rng) -> SAState for the starting state.
    seed : RNG seed for reproducibility.
    verbose : print periodic progress.
    """
    rng = np.random.default_rng(seed)

    def effective_energy(state):
        e = energy(state, K_ik, K_trg, lambda_repel, lat_ik, lat_trg, lambda_lateral)
        if penalty_fn is not None and gamma_penalty != 0.0:
            e += gamma_penalty * penalty_fn(state)
        return e

    if init_fn is not None:
        current = init_fn(rng)
    else:
        current = initialize_state(ik_leaves, ik_internals,
                                   trg_leaves, trg_internals, rng)
    current_energy = effective_energy(current.state)

    best = current.copy()
    best_energy = current_energy

    history = [current_energy]

    while T > T_min:
        for _ in range(iters_per_temp):
            candidate = propose_neighbor(current, rng, kinematic_filter)
            new_energy = effective_energy(candidate.state)
            delta_E = new_energy - current_energy

            if delta_E < 0:
                accept = True
            else:
                # exp(-delta/T); guard against overflow for large delta/T.
                P = np.exp(-delta_E / T)
                accept = rng.random() < P

            if accept:
                current = candidate
                current_energy = new_energy
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy

        history.append(current_energy)
        if verbose:
            print(f"T={T:.5f}  E={current_energy:.6f}  best={best_energy:.6f}")

        T *= alpha

    return best.state, best_energy, history


def simulated_annealing_restarts(K_ik, K_trg,
                                 ik_leaves, ik_internals,
                                 trg_leaves, trg_internals,
                                 n_restarts=2,
                                 seed=None,
                                 verbose=False,
                                 **sa_kwargs):
    """Run SA `n_restarts` times from independent seeds; keep the best.

    The search space here is tiny (a 17-node injection), so multiple random
    restarts buy more robustness against local minima than any single clever
    cooling schedule. Returns the same (best_state, best_energy, history) tuple
    as simulated_annealing(), for the single best restart.

    Seeding: a single fixed seed would make every restart identical, which is
    pointless. We use a SeedSequence to spawn `n_restarts` child streams that
    are mutually distinct yet fully reproducible across runs.
    """
    child_seeds = np.random.SeedSequence(seed).spawn(n_restarts)

    best_state = None
    best_energy = np.inf
    best_history = None

    for i, child in enumerate(child_seeds):
        state, e, hist = simulated_annealing(
            K_ik, K_trg,
            ik_leaves, ik_internals,
            trg_leaves, trg_internals,
            seed=child,
            verbose=False,
            **sa_kwargs,
        )
        if e < best_energy:
            best_state, best_energy, best_history = state, e, hist
        if verbose:
            print(f"restart {i + 1:>2}/{n_restarts}: "
                  f"E={e:.6f}  best={best_energy:.6f}")

    return best_state, best_energy, best_history


# --------------------------------------------------------------------------- #
# Convenience: run straight from normalized distance matrices
# --------------------------------------------------------------------------- #

def run_from_distance_matrices(D_ik, D_trg, **sa_kwargs):
    """End-to-end helper: take two normalized [0,1] distance matrices, build
    affinities, classify nodes, and run SA. Returns the SA result tuple.

    Node degrees are inferred from nonzero off-diagonal entries of each
    distance matrix, so a 0 distance is treated as "not adjacent". If your
    distance matrices are fully dense geodesics (no zeros off-diagonal), pass
    explicit leaf/internal arrays via sa_kwargs instead.
    """
    K_ik = build_affinity(D_ik)
    K_trg = build_affinity(D_trg)

    if "ik_leaves" not in sa_kwargs:
        ik_leaves, ik_internals = classify_nodes(D_ik)
        sa_kwargs["ik_leaves"], sa_kwargs["ik_internals"] = ik_leaves, ik_internals
    if "trg_leaves" not in sa_kwargs:
        trg_leaves, trg_internals = classify_nodes(D_trg)
        sa_kwargs["trg_leaves"], sa_kwargs["trg_internals"] = trg_leaves, trg_internals

    return simulated_annealing(K_ik, K_trg, **sa_kwargs)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Small synthetic example: 5 IK nodes, 7 Target nodes.
    # IK: leaves {0, 4}, internals {1, 2, 3}
    # Target: leaves {0, 4, 5}, internals {1, 2, 3, 6}
    n_ik, n_trg = 5, 7
    K_ik = build_affinity(rng.random((n_ik, n_ik)) * 0.5)
    K_ik = (K_ik + K_ik.T) / 2
    K_trg = build_affinity(rng.random((n_trg, n_trg)) * 0.5)
    K_trg = (K_trg + K_trg.T) / 2

    ik_leaves, ik_internals = np.array([0, 4]), np.array([1, 2, 3])
    trg_leaves, trg_internals = np.array([0, 4, 5]), np.array([1, 2, 3, 6])

    best_state, best_E, hist = simulated_annealing(
        K_ik, K_trg,
        ik_leaves, ik_internals,
        trg_leaves, trg_internals,
        T=1.0, alpha=0.995, T_min=1e-4,
        seed=42, verbose=False,
    )

    print("Best mapping (IK node -> Target node):")
    for ik_node, trg_node in enumerate(best_state):
        kind = "leaf" if ik_node in ik_leaves else "internal"
        print(f"  IK {ik_node} ({kind}) -> Target {trg_node}")
    print(f"\nFinal energy: {best_E:.6f}")
    print(f"Start energy: {hist[0]:.6f}")
    print(f"Improvement:  {hist[0] - best_E:.6f}")

    # Validate constraints on the output.
    assert len(set(best_state)) == n_ik, "Mapping must be injective."
    assert all(best_state[i] in trg_leaves for i in ik_leaves), "Leaf->leaf violated."
    assert all(best_state[i] in trg_internals for i in ik_internals), "Internal->internal violated."
    print("\nAll topological constraints satisfied.")
