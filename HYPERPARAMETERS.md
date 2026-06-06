# Hyperparameter Reference

All tunable values and hardcoded thresholds in the SA skeleton-matching pipeline.
Ranked within each section by impact on result quality.

---

## 1. Affinity Kernel

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `sigma` (σ) | `simulated_annealing.py:27` `build_affinity(sigma=0.1)` | `0.1` | **`1`** (`SIGMA` in `main.py`) | Kernel bandwidth in `K = e^{-D/σ}`. With D max-normalized to [0,1], controls near/far contrast in the affinity matrix. Smaller σ → sharper contrast. The entire QAP signal sensitivity lives here. |

---

## 2. Repellent Term

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `lambda_repel` (λ) | `simulated_annealing.py` `energy(lambda_repel=0.0)` / `simulated_annealing(lambda_repel=0.0)` | `0.0` | **`0.2`** (`LAMBDA_REPEL` in `main.py`) | Weight of the repellent term `λ · Σ (1 − K_ik) · K_trg`. Penalises IK-distant pairs landing on target-close nodes. 0.0 = plain QAP; raise if the solver clusters joints onto adjacent spine/torso bones. |

> Full energy: `E = -Σ K_ik · K_trg  +  λ · Σ (1 − K_ik) · K_trg`  
> Equivalently: `E = -(1+λ) · Σ K_ik · K_trg  +  λ · Σ K_trg[Π(i),Π(j)]`  
> The second form shows the repellent reduces to minimising total pairwise target affinity — maximising spread of assignments across the target skeleton.

---

## 3. Cooling Schedule

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `T` | `simulated_annealing.py` `simulated_annealing(T=1.0)` | `1.0` | `1.0` | Initial temperature |
| `alpha` | `simulated_annealing.py` `simulated_annealing(alpha=0.999)` | `0.999` | `0.99` | Geometric cooling rate — `T *= alpha` after every temperature step |
| `T_min` | `simulated_annealing.py` `simulated_annealing(T_min=1e-7)` | `1e-7` | `1e-5` | Loop terminates when `T < T_min` |
| `iters_per_temp` | `simulated_annealing.py` `simulated_annealing(iters_per_temp=1)` | `1` | `100` | SA proposals attempted at each temperature before cooling |

> Total proposals per restart ≈ `iters_per_temp × log(T_min / T) / log(alpha)`  
> With active values: `100 × log(1e-5) / log(0.99) ≈ 114,000` steps/restart.

---

## 4. Multi-Restart

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `n_restarts` | `simulated_annealing.py` `simulated_annealing_restarts(n_restarts=2)` | `2` | `7` | Independent SA runs from different seeds; the global best is returned |
| `seed` | `main.py` `run_sa_pipeline(seed=91)` | `91` | `91` | Master seed fed to `SeedSequence.spawn()` to generate per-restart child seeds |

---

## 5. Neighbor Proposal

| Parameter | Defined in | Default | Role |
|-----------|-----------|---------|------|
| `max_tries` | `simulated_annealing.py` `propose_neighbor(max_tries=32)` | `32` | Max proposal attempts per step before falling back to a no-op copy. Guards against the kinematic filter rejecting everything in a tight feasible region. |

---

## 6. Kinematic Ancestor Filter

Hardcoded constraints in `main.py:build_hierarchy_constraints`. Not exposed as function parameters.

| Constraint | Code | Role |
|------------|------|------|
| Proper ancestry | `state[parent_pos] in trg_ancestors[state[child_pos]]` | For every IK edge (parent→child), the target node assigned to the IK parent must be a **proper ancestor** of the target node assigned to the IK child — i.e. it must lie on the root-to-child path in the target tree. Rules out cross-branch assignments that would pass a depth-only check. |

> `trg_ancestors[p]` = frozenset of target positions on the path from target root to `p`, **excluding p itself**. Precomputed once before SA starts.  
> `trg_depth` is still precomputed and used as a tiebreaker in `init_fn` (prefer shallowest valid descendant) but no longer drives the filter itself.

---

## 7. Pruning Thresholds

Hardcoded in `graph_modification.py`. Applied before SA runs; determines what the target skeleton looks like.

| Threshold | Value | Role |
|-----------|-------|------|
| Fan-out detection | `out_degree > 2` | A target node with 3+ children is a candidate anchor |
| Terminal-child gate | `terminal_child_count >= 2` | Anchor is registered only if ≥ 2 of its children lead into purely terminal subtrees. Distinguishes end-effectors (wrist with fingers) from structural joints (thorax, pelvis). |

---

## 8. IK Skeleton — Bone Lengths

All 16 values in `H36M_BONE_LENGTHS` (`skeleton.py`) feed into Floyd-Warshall and shape `D_ik` / `K_ik`. Absolute scale is removed by max-normalization; **relative ratios matter**.

| Bone | Child joint | Length (m) |
|------|-------------|-----------|
| Hip → RHip | 1 | 0.13 |
| RHip → RKnee | 2 | 0.44 |
| RKnee → RFoot | 3 | 0.45 |
| Hip → LHip | 4 | 0.13 |
| LHip → LKnee | 5 | 0.44 |
| LKnee → LFoot | 6 | 0.45 |
| Hip → Spine | 7 | 0.23 |
| Spine → Thorax | 8 | 0.23 |
| Thorax → Neck | 9 | 0.12 |
| Neck → Head | 10 | 0.11 |
| Thorax → LShoulder | 11 | 0.15 |
| LShoulder → LElbow | 12 | 0.28 |
| LElbow → LWrist | 13 | 0.25 |
| Thorax → RShoulder | 14 | 0.15 |
| RShoulder → RElbow | 15 | 0.28 |
| RElbow → RWrist | 16 | 0.25 |

---

## 9. Visualization (non-algorithmic)

| Parameter | File | Value | Role |
|-----------|------|-------|------|
| `box` | `visualize_mapping.py` | `500.0` | Bounding box size for `fit_positions()` scaling |
| `PROJ_AXES` | `visualize_mapping.py` | `(0, 1)` | World axes projected onto the 2D canvas (X, Y) |
| `FLIP_Y` | `visualize_mapping.py` | `True` | Negates Y to correct for Cytoscape's downward-Y vs GLB's upward-Y |
| `spacingFactor` | `visualize_graph.py` | `1.75` | Breadthfirst layout node spacing |
| Port | `visualize_mapping.py` | `8051` | Dash server port for the mapping viewer |
| Port | `visualize_graph.py` | `8050` | Dash server port for the graph viewer |

---

## Impact Ranking

| Rank | Parameter | Why it matters |
|------|-----------|---------------|
| 1 | `sigma` | Wrong value collapses the entire QAP signal — the affinity matrix becomes flat |
| 2 | `lambda_repel` | Too low → spine clustering persists; too high → repulsion overwhelms attraction and scatters close joints |
| 3 | `alpha` + `T_min` | Controls total search time; freeze too early → stuck in a local minimum |
| 4 | `iters_per_temp` | Trades exploration breadth vs cooling granularity |
| 5 | `n_restarts` | Cheapest robustness lever for a small problem |
| 6 | Pruning thresholds | Determine the target skeleton shape before any SA runs |
| 7 | `max_tries` | Only binding when the kinematic filter is very restrictive |
| 8 | Bone lengths | Affect `D_ik` shape; relative ratios matter more than absolute values |
