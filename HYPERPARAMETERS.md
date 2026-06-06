# Hyperparameter Reference

All tunable values and hardcoded thresholds in the SA skeleton-matching pipeline.
Ranked within each section by impact on result quality.

---

## 1. Affinity Kernel

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `sigma` (σ) | `simulated_annealing.py` `build_affinity(sigma=0.1)` | `0.1` | **`1`** (`SIGMA` in `main.py`) | Kernel bandwidth in `K = e^{-D/σ}`. With D max-normalized to [0,1], controls near/far contrast in the affinity matrix. Smaller σ → sharper contrast. The entire QAP signal sensitivity lives here. |

---

## 2. Mismatch Penalty

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `lambda_repel` (λ) | `simulated_annealing.py` `energy(lambda_repel=0.0)` / `simulated_annealing(lambda_repel=0.0)` | `0.0` | **`0.3`** (`LAMBDA_REPEL` in `main.py`) | Weight of the symmetric mismatch term `λ · Σ \|K_ik − K_trg\|`. Penalises any affinity asymmetry regardless of direction: far-IK-on-close-target (spine clustering) and close-IK-on-far-target (chain skipping) both cost the same λ per pair. 0.0 = plain QAP. |

> Full energy: `E = -Σ K_ik · K_trg  +  λ · Σ |K_ik − K_trg[Π(i),Π(j)]|`
>
> Four-quadrant cost (K ≈ 0 or 1 at small σ):
>
> |  | target-close | target-far |
> |--|--|--|
> | **IK-close** | \|1−1\| ≈ 0 ✓ | \|1−0\| = 1 penalised ✓ |
> | **IK-far** | \|0−1\| = 1 penalised ✓ | \|0−0\| ≈ 0 ✓ |
>
> **Tuning:** raise λ if spine clustering or leg chain-skipping persists; lower if close joints scatter.

---

## 3. Ancestor Ordering — Soft Penalty

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `gamma_penalty` (γ) | `simulated_annealing.py` `simulated_annealing(gamma_penalty=0.0)` | `0.0` | **`5.0`** (`GAMMA_PENALTY` in `main.py`) | Scale factor for `ancestor_penalty_fn(state)`, which counts IK edges whose ancestor constraint is violated. Each violated edge adds `gamma_penalty` to the effective energy. |

> Full effective energy: `E_eff = E_QAP + γ · violations(state)`  
> Where `violations` = number of IK edges where the assigned target-parent is **not** a proper ancestor of the assigned target-child in the target tree.  
>
> **Why soft, not hard:** A hard gate on proposals kills SA exploration — leaf/internal swaps almost always land on a different target branch, burning through `max_tries` and producing no-op steps. The soft penalty lets the SA cross violations at high T and converge to valid solutions at low T, which is the intended SA behaviour.  
>
> **Tuning:** `gamma_penalty` should be large enough that one violation outweighs a typical per-pair QAP gain (~1 energy unit with σ=1). Start at 5.0; raise if the final mapping still has cross-branch assignments.

---

## 4. Cooling Schedule

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `T` | `simulated_annealing.py` `simulated_annealing(T=1.0)` | `1.0` | `1.0` | Initial temperature |
| `alpha` | `simulated_annealing.py` `simulated_annealing(alpha=0.999)` | `0.999` | `0.99` | Geometric cooling rate — `T *= alpha` after every temperature step |
| `T_min` | `simulated_annealing.py` `simulated_annealing(T_min=1e-7)` | `1e-7` | `1e-5` | Loop terminates when `T < T_min` |
| `iters_per_temp` | `simulated_annealing.py` `simulated_annealing(iters_per_temp=1)` | `1` | `100` | SA proposals attempted at each temperature before cooling |

> Total proposals per restart ≈ `iters_per_temp × log(T_min / T) / log(alpha)`  
> With active values: `100 × log(1e-5) / log(0.99) ≈ 114,000` steps/restart.

---

## 5. Multi-Restart

| Parameter | Defined in | Default | Active value | Role |
|-----------|-----------|---------|-------------|------|
| `n_restarts` | `simulated_annealing.py` `simulated_annealing_restarts(n_restarts=2)` | `2` | `7` | Independent SA runs from different seeds; the global best is returned |
| `seed` | `main.py` `run_sa_pipeline(seed=91)` | `91` | `91` | Master seed fed to `SeedSequence.spawn()` to generate per-restart child seeds |

---

## 6. Neighbor Proposal

| Parameter | Defined in | Default | Role |
|-----------|-----------|---------|------|
| `max_tries` | `simulated_annealing.py` `propose_neighbor(max_tries=32)` | `32` | Max proposal attempts before falling back to a no-op. Relevant only for hard `kinematic_filter` use; with the ancestor constraint moved to a soft penalty, this is rarely the binding limit. |

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
| 2 | `lambda_repel` | Too low → spine clustering or chain-skipping persists; too high → mismatch penalty overwhelms attraction and scatters close joints |
| 3 | `gamma_penalty` | Too low → cross-branch assignments survive to the final result; too high → SA can't explore at high T and freezes near the initial state |
| 4 | `alpha` + `T_min` | Controls total search time; freeze too early → stuck in a local minimum |
| 5 | `iters_per_temp` | Trades exploration breadth vs cooling granularity |
| 6 | `n_restarts` | Cheapest robustness lever for a small problem |
| 7 | Pruning thresholds | Determine the target skeleton shape before any SA runs |
| 8 | Bone lengths | Affect `D_ik` shape; relative ratios matter more than absolute values |
| 9 | `max_tries` | Only binding with a hard kinematic_filter; currently not the bottleneck |
