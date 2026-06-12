"""
SA Skeleton Correspondence — Evaluation Suite.

Three experiments:
  1. Ablation  — what each component contributes (5 configs x all annotated GLBs)
  2. Accuracy  — full method on all annotated GLBs, per-file + aggregate
  3. Robustness — stability across 20 seeds on the first annotated GLB

Run:  python src/evaluate.py
Results saved to results/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import csv
import json
import glob
import numpy as np
import networkx as nx

from main import (run_sa_pipeline, build_ik_rig, build_hierarchy_constraints,
                  graph_to_normalized_matrix, SIGMA, LAMBDA_REPEL, GAMMA_PENALTY)
from simulated_annealing import build_affinity, energy
from skeleton import H36M_NAMES

# ── module-level paths ────────────────────────────────────────────────────────

ASSETS_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results'))

# ── ablation configuration table ──────────────────────────────────────────────

ABLATION_CONFIGS = [
    {'name': 'Plain QAP',      'lambda_repel': 0.0, 'gamma_penalty': 0.0, 'n_restarts': 1},
    {'name': 'Lambda only',    'lambda_repel': 0.5, 'gamma_penalty': 0.0, 'n_restarts': 1},
    {'name': 'Gamma only',     'lambda_repel': 0.0, 'gamma_penalty': 3.0, 'n_restarts': 1},
    {'name': 'Both penalties', 'lambda_repel': 0.5, 'gamma_penalty': 3.0, 'n_restarts': 1},
    {'name': 'Full method',    'lambda_repel': 0.5, 'gamma_penalty': 3.0, 'n_restarts': 7},
]

# ── limb definitions (H36M joint names) ──────────────────────────────────────

LIMBS = {
    'LEFT_ARM':   ['LShoulder', 'LElbow', 'LWrist'],
    'RIGHT_ARM':  ['RShoulder', 'RElbow', 'RWrist'],
    'LEFT_LEG':   ['LHip', 'LKnee', 'LFoot'],
    'RIGHT_LEG':  ['RHip', 'RKnee', 'RFoot'],
    'SPINE_HEAD': ['Hip', 'Spine', 'Thorax', 'Neck', 'Head'],
}


# ── 2a. load_annotations ─────────────────────────────────────────────────────

def load_annotations(annotations_path: str) -> dict:
    """Read annotations.json and return the parsed dict."""
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(
            "annotations.json not found. Run src/annotator.py first to create ground truth."
        )
    with open(annotations_path) as f:
        return json.load(f)


# ── 2b. build_ground_truth ───────────────────────────────────────────────────

def build_ground_truth(glb_filename: str, pruned_G, trg_ordered: list,
                       annotations: dict) -> dict:
    """Return {ik_pos_int: trg_ordered_position_int} for annotated joints."""
    if glb_filename not in annotations:
        raise KeyError(f"No annotation found for {glb_filename}. Annotate it first.")

    assignments = annotations[glb_filename].get('assignments', {})
    n = len(assignments)
    if n < 17:
        print(f"WARNING: {glb_filename} has only {n}/17 joints annotated.")

    name_to_node = {data.get('name', ''): node_id
                    for node_id, data in pruned_G.nodes(data=True)}

    ground_truth = {}
    for ik_idx_str, bone_name in assignments.items():
        ik_pos = int(ik_idx_str)
        node_id = name_to_node.get(bone_name)
        if node_id is None:
            print(f"WARNING: bone '{bone_name}' not found in pruned graph — skipping.")
            continue
        ground_truth[ik_pos] = trg_ordered.index(node_id)

    return ground_truth


# ── 2c. hop_distance ─────────────────────────────────────────────────────────

def hop_distance(pos_a: int, pos_b: int, pruned_G, trg_ordered: list) -> int:
    """Shortest undirected hop count between two positions in the pruned graph."""
    if pos_a == pos_b:
        return 0
    node_a = trg_ordered[pos_a]
    node_b = trg_ordered[pos_b]
    UG = pruned_G.to_undirected()
    try:
        return nx.shortest_path_length(UG, node_a, node_b)
    except nx.NetworkXNoPath:
        return 999


# ── 2d. detect_symmetry_flip ─────────────────────────────────────────────────

def detect_symmetry_flip(best_state, ik_ordered, pruned_G, trg_ordered) -> bool:
    """True if >= 4 of the 6 left IK joints are assigned to a 'Right' target bone."""
    left_positions = [
        i for i, n in enumerate(ik_ordered)
        if H36M_NAMES.get(n, '').startswith('L')
    ]
    flipped = 0
    for i in left_positions:
        trg_pos = int(best_state[i])
        node_id = trg_ordered[trg_pos]
        name = pruned_G.nodes[node_id].get('name', '')
        if 'Right' in name:
            flipped += 1
    return flipped >= 4


# ── 2e. compute_metrics ───────────────────────────────────────────────────────

def compute_metrics(best_state, ik_ordered, pruned_G, trg_ordered,
                    ground_truth: dict, K_ik, K_trg, G_ik) -> dict:
    """Compute all evaluation metrics for one SA result against ground truth."""
    n_annotated = len(ground_truth)

    # Hop distances per annotated joint
    hop_list = []
    hop_by_joint = {}
    for ik_pos, gt_trg_pos in ground_truth.items():
        pred_trg_pos = int(best_state[ik_pos])
        hop = hop_distance(pred_trg_pos, gt_trg_pos, pruned_G, trg_ordered)
        hop_list.append(hop)
        joint_name = H36M_NAMES[ik_ordered[ik_pos]]
        hop_by_joint[joint_name] = hop

    exact_acc  = sum(1 for h in hop_list if h == 0) / n_annotated * 100
    near_acc   = sum(1 for h in hop_list if h <= 1) / n_annotated * 100
    wrong_limb = sum(1 for h in hop_list if h >= 3) / n_annotated * 100

    correct_limbs = sum(
        1 for joints in LIMBS.values()
        if all(j in hop_by_joint and hop_by_joint[j] <= 1 for j in joints)
    )
    limb_acc = correct_limbs / 5 * 100

    ancestor_penalty_fn, _ = build_hierarchy_constraints(
        G_ik, ik_ordered, pruned_G, trg_ordered)
    violation_count = ancestor_penalty_fn(best_state)
    avr = violation_count / 16 * 100

    flip = detect_symmetry_flip(best_state, ik_ordered, pruned_G, trg_ordered)

    qap_energy = energy(best_state, K_ik, K_trg, lambda_repel=0.0)

    return {
        'exact_acc':  exact_acc,
        'near_acc':   near_acc,
        'wrong_limb': wrong_limb,
        'limb_acc':   limb_acc,
        'avr':        avr,
        'flip':       flip,
        'qap_energy': qap_energy,
        'hop_list':   hop_list,
        'hop_by_joint': hop_by_joint,
    }


# ── shared: run pipeline + rebuild affinities ─────────────────────────────────

def _run_and_build_affinities(glb_path: str, sigma: float,
                               lambda_repel: float, gamma_penalty: float,
                               n_restarts: int, seed: int):
    """Run the SA pipeline and return (R, K_ik, K_trg)."""
    R = run_sa_pipeline(
        glb_path=glb_path,
        n_restarts=n_restarts,
        seed=seed,
        sigma=sigma,
        lambda_repel=lambda_repel,
        gamma_penalty=gamma_penalty,
        verbose=False,
    )
    D_ik  = graph_to_normalized_matrix(R['G_ik'],    R['ik_ordered'])
    D_trg = graph_to_normalized_matrix(R['pruned_G'], R['trg_ordered'])
    K_ik  = build_affinity(D_ik,  sigma=sigma)
    K_trg = build_affinity(D_trg, sigma=sigma)
    return R, K_ik, K_trg


# ── Experiment 1 — Ablation ───────────────────────────────────────────────────

def run_ablation(annotations_path: str,
                 assets_dir: str = ASSETS_DIR,
                 sigma: float = 0.2,
                 seed: int = 91) -> list:
    """Ablation study: 5 configurations x all annotated GLBs.

    Parameters exposed so you can re-run with e.g. sigma=0.3 without touching
    the config table. Returns a list of 5 averaged-metric dicts.
    """
    annotations = load_annotations(annotations_path)
    glb_files   = list(annotations.keys())
    print(f"\n=== ABLATION ({len(glb_files)} file(s), sigma={sigma}) ===")

    all_results = []

    for cfg in ABLATION_CONFIGS:
        cfg_metrics = []
        print(f"\n  Config: {cfg['name']}")
        for glb_filename in glb_files:
            glb_path = os.path.join(assets_dir, glb_filename)
            try:
                R, K_ik, K_trg = _run_and_build_affinities(
                    glb_path, sigma=sigma,
                    lambda_repel=cfg['lambda_repel'],
                    gamma_penalty=cfg['gamma_penalty'],
                    n_restarts=cfg['n_restarts'],
                    seed=seed,
                )
                ground_truth = build_ground_truth(
                    glb_filename, R['pruned_G'], R['trg_ordered'], annotations)
                m = compute_metrics(
                    R['best_state'], R['ik_ordered'], R['pruned_G'], R['trg_ordered'],
                    ground_truth, K_ik, K_trg, R['G_ik'])
                cfg_metrics.append(m)
                print(f"    {glb_filename}: exact={m['exact_acc']:.1f}%  "
                      f"limb={m['limb_acc']:.1f}%  E={m['qap_energy']:.4f}")
            except Exception as e:
                print(f"    WARNING: {glb_filename} skipped — {e}")

        if not cfg_metrics:
            continue

        avg = {
            'config':     cfg['name'],
            'exact_acc':  float(np.mean([m['exact_acc']           for m in cfg_metrics])),
            'near_acc':   float(np.mean([m['near_acc']            for m in cfg_metrics])),
            'wrong_limb': float(np.mean([m['wrong_limb']          for m in cfg_metrics])),
            'limb_acc':   float(np.mean([m['limb_acc']            for m in cfg_metrics])),
            'avr':        float(np.mean([m['avr']                 for m in cfg_metrics])),
            'flip_rate':  float(np.mean([float(m['flip'])         for m in cfg_metrics])) * 100,
            'qap_energy': float(np.mean([m['qap_energy']          for m in cfg_metrics])),
        }
        all_results.append(avg)

    # Print summary table
    header = f"\n{'Config':<18} {'Exact%':>8} {'Limb%':>8} {'AVR%':>8} {'Flip%':>8} {'QAP E':>10}"
    print(header)
    print("-" * 64)
    for r in all_results:
        print(f"{r['config']:<18} {r['exact_acc']:>8.1f} {r['limb_acc']:>8.1f} "
              f"{r['avr']:>8.1f} {r['flip_rate']:>8.1f} {r['qap_energy']:>10.4f}")

    # Save CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _path = os.path.join(RESULTS_DIR, 'ablation.csv')
    with open(_path, 'w', newline='') as f:
        fields = ['config', 'exact_acc', 'near_acc', 'wrong_limb',
                  'limb_acc', 'avr', 'flip_rate', 'qap_energy']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {_path}")

    return all_results


# ── Experiment 2 — Accuracy Evaluation ───────────────────────────────────────

def run_accuracy_eval(annotations_path: str,
                      assets_dir: str = ASSETS_DIR,
                      sigma: float = 0.2,
                      lambda_repel: float = 0.5,
                      gamma_penalty: float = 3.0,
                      n_restarts: int = 7,
                      seed: int = 91) -> dict:
    """Full method accuracy on all annotated GLBs.

    All SA parameters exposed for easy parameter sweeps.
    Returns an aggregation dict (means, stds, flip stats).
    """
    annotations = load_annotations(annotations_path)
    glb_files   = list(annotations.keys())
    print(f"\n=== ACCURACY EVAL ({len(glb_files)} file(s), "
          f"σ={sigma} λ={lambda_repel} γ={gamma_penalty} restarts={n_restarts}) ===")

    per_file = []

    for glb_filename in glb_files:
        glb_path = os.path.join(assets_dir, glb_filename)
        try:
            R, K_ik, K_trg = _run_and_build_affinities(
                glb_path, sigma=sigma,
                lambda_repel=lambda_repel,
                gamma_penalty=gamma_penalty,
                n_restarts=n_restarts,
                seed=seed,
            )
            ground_truth = build_ground_truth(
                glb_filename, R['pruned_G'], R['trg_ordered'], annotations)
            m = compute_metrics(
                R['best_state'], R['ik_ordered'], R['pruned_G'], R['trg_ordered'],
                ground_truth, K_ik, K_trg, R['G_ik'])
            m['file'] = glb_filename
            per_file.append(m)
        except Exception as e:
            print(f"WARNING: {glb_filename} skipped — {e}")

    if not per_file:
        print("No files evaluated.")
        return {}

    # Per-file table
    print(f"\n{'File':<36} {'Exact%':>7} {'Limb%':>7} {'Wrong%':>7} "
          f"{'AVR%':>7} {'Flip':>5} {'QAP E':>10}")
    print("-" * 82)
    for m in per_file:
        flip_str = 'YES' if m['flip'] else 'no'
        print(f"{m['file']:<36} {m['exact_acc']:>7.1f} {m['limb_acc']:>7.1f} "
              f"{m['wrong_limb']:>7.1f} {m['avr']:>7.1f} {flip_str:>5} "
              f"{m['qap_energy']:>10.4f}")

    # Aggregate
    N = len(per_file)
    keys = ['exact_acc', 'near_acc', 'wrong_limb', 'limb_acc', 'avr', 'qap_energy']
    agg = {k: {'mean': float(np.mean([m[k] for m in per_file])),
               'std':  float(np.std( [m[k] for m in per_file]))}
           for k in keys}

    flipped_files   = [m for m in per_file if m['flip']]
    unflipped_files = [m for m in per_file if not m['flip']]
    agg['flip_rate']            = len(flipped_files) / N * 100
    agg['flip_excluded_exact']  = (float(np.mean([m['exact_acc'] for m in unflipped_files]))
                                   if unflipped_files else float('nan'))
    agg['flip_excluded_limb']   = (float(np.mean([m['limb_acc'] for m in unflipped_files]))
                                   if unflipped_files else float('nan'))

    print(f"\n=== AGGREGATE ({N} file(s)) ===")
    print(f"Exact accuracy:          {agg['exact_acc']['mean']:>6.1f} ± {agg['exact_acc']['std']:.1f} %")
    print(f"Near accuracy (hop<=1):  {agg['near_acc']['mean']:>6.1f} ± {agg['near_acc']['std']:.1f} %")
    print(f"Wrong limb rate:         {agg['wrong_limb']['mean']:>6.1f} ± {agg['wrong_limb']['std']:.1f} %")
    print(f"Limb accuracy:           {agg['limb_acc']['mean']:>6.1f} ± {agg['limb_acc']['std']:.1f} %")
    print(f"AVR:                     {agg['avr']['mean']:>6.1f} ± {agg['avr']['std']:.1f} %")
    print(f"Flip rate:               {agg['flip_rate']:>6.1f} %")
    print(f"Flip-excluded exact acc: {agg['flip_excluded_exact']:>6.1f} %")
    print(f"Flip-excluded limb acc:  {agg['flip_excluded_limb']:>6.1f} %")

    # Save CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    acc_path = os.path.join(RESULTS_DIR, 'accuracy.csv')
    with open(acc_path, 'w', newline='') as f:
        fields = ['file', 'exact_acc', 'near_acc', 'wrong_limb',
                  'limb_acc', 'avr', 'flip', 'qap_energy']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(per_file)
    print(f"\nSaved {acc_path}")

    sum_path = os.path.join(RESULTS_DIR, 'accuracy_summary.csv')
    with open(sum_path, 'w', newline='') as f:
        fields = [
            'n_files',
            'exact_acc_mean', 'exact_acc_std',
            'near_acc_mean',  'near_acc_std',
            'wrong_limb_mean', 'wrong_limb_std',
            'limb_acc_mean',  'limb_acc_std',
            'avr_mean', 'avr_std',
            'flip_rate',
            'flip_excluded_exact', 'flip_excluded_limb',
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            'n_files': N,
            'exact_acc_mean': agg['exact_acc']['mean'],
            'exact_acc_std':  agg['exact_acc']['std'],
            'near_acc_mean':  agg['near_acc']['mean'],
            'near_acc_std':   agg['near_acc']['std'],
            'wrong_limb_mean': agg['wrong_limb']['mean'],
            'wrong_limb_std':  agg['wrong_limb']['std'],
            'limb_acc_mean':  agg['limb_acc']['mean'],
            'limb_acc_std':   agg['limb_acc']['std'],
            'avr_mean': agg['avr']['mean'],
            'avr_std':  agg['avr']['std'],
            'flip_rate': agg['flip_rate'],
            'flip_excluded_exact': agg['flip_excluded_exact'],
            'flip_excluded_limb':  agg['flip_excluded_limb'],
        })
    print(f"Saved {sum_path}")

    return agg


# ── Experiment 3 — Robustness ─────────────────────────────────────────────────

def run_robustness(annotations_path: str,
                   n_runs: int = 20,
                   assets_dir: str = ASSETS_DIR,
                   sigma: float = 0.2,
                   lambda_repel: float = 0.5,
                   gamma_penalty: float = 3.0,
                   n_restarts: int = 7) -> dict:
    """Stability test: run full method n_runs times (seeds 0..n_runs-1).

    Uses the first annotated GLB as the test rig. All SA parameters exposed
    so you can sweep them independently of the main accuracy experiment.
    Returns a summary dict with means, stds, flip count, and best-run history.
    """
    annotations = load_annotations(annotations_path)
    glb_filename = next(iter(annotations))
    glb_path     = os.path.join(assets_dir, glb_filename)
    print(f"\n=== ROBUSTNESS ({n_runs} runs, file={glb_filename}, "
          f"σ={sigma} λ={lambda_repel} γ={gamma_penalty} restarts={n_restarts}) ===")

    # Build ground truth once (doesn't change across seeds)
    _R0, _, _ = _run_and_build_affinities(
        glb_path, sigma=sigma,
        lambda_repel=lambda_repel, gamma_penalty=gamma_penalty,
        n_restarts=n_restarts, seed=0,
    )
    ground_truth = build_ground_truth(
        glb_filename, _R0['pruned_G'], _R0['trg_ordered'], annotations)

    per_run = []
    best_energy_seen = np.inf
    best_history     = None

    print(f"\n{'Run':>4} {'Seed':>5} {'Exact%':>8} {'Limb%':>8} {'QAP E':>10} {'Flip':>5}")
    print("-" * 44)

    for run_idx in range(n_runs):
        seed = run_idx
        try:
            R, K_ik, K_trg = _run_and_build_affinities(
                glb_path, sigma=sigma,
                lambda_repel=lambda_repel, gamma_penalty=gamma_penalty,
                n_restarts=n_restarts, seed=seed,
            )
            m = compute_metrics(
                R['best_state'], R['ik_ordered'], R['pruned_G'], R['trg_ordered'],
                ground_truth, K_ik, K_trg, R['G_ik'])

            if R['best_energy'] < best_energy_seen:
                best_energy_seen = R['best_energy']
                best_history     = R['history']

            row = {
                'run':        run_idx,
                'seed':       seed,
                'exact_acc':  m['exact_acc'],
                'limb_acc':   m['limb_acc'],
                'qap_energy': m['qap_energy'],
                'flip':       m['flip'],
            }
            per_run.append(row)
            flip_str = 'YES' if m['flip'] else 'no'
            print(f"{run_idx:>4} {seed:>5} {m['exact_acc']:>8.1f} {m['limb_acc']:>8.1f} "
                  f"{m['qap_energy']:>10.4f} {flip_str:>5}")

        except Exception as e:
            print(f"  Run {run_idx} (seed={seed}) failed — {e}")

    if not per_run:
        print("No runs completed.")
        return {}

    N = len(per_run)
    summary = {
        'n_runs':           N,
        'qap_energy_mean':  float(np.mean([r['qap_energy'] for r in per_run])),
        'qap_energy_std':   float(np.std( [r['qap_energy'] for r in per_run])),
        'exact_acc_mean':   float(np.mean([r['exact_acc']  for r in per_run])),
        'exact_acc_std':    float(np.std( [r['exact_acc']  for r in per_run])),
        'limb_acc_mean':    float(np.mean([r['limb_acc']   for r in per_run])),
        'limb_acc_std':     float(np.std( [r['limb_acc']   for r in per_run])),
        'n_flips':          sum(1 for r in per_run if r['flip']),
    }

    print(f"\n=== ROBUSTNESS SUMMARY ({N} runs) ===")
    print(f"QAP energy:     {summary['qap_energy_mean']:.4f} ± {summary['qap_energy_std']:.4f}")
    print(f"Exact accuracy: {summary['exact_acc_mean']:.1f} ± {summary['exact_acc_std']:.1f} %")
    print(f"Limb accuracy:  {summary['limb_acc_mean']:.1f} ± {summary['limb_acc_std']:.1f} %")
    print(f"Flip rate:      {summary['n_flips']} / {N} runs")

    # Save CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rob_path = os.path.join(RESULTS_DIR, 'robustness.csv')
    with open(rob_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'seed', 'exact_acc',
                                               'limb_acc', 'qap_energy', 'flip'])
        writer.writeheader()
        writer.writerows(per_run)
    print(f"\nSaved {rob_path}")

    if best_history is not None:
        conv_path = os.path.join(RESULTS_DIR, 'convergence.csv')
        with open(conv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'energy'])
            writer.writeheader()
            writer.writerows({'step': i, 'energy': e}
                             for i, e in enumerate(best_history))
        print(f"Saved {conv_path}")

    return summary


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ANNOTATIONS_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'assets', 'annotations.json')
    ASSETS_DIR_MAIN  = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'assets'))

    if not os.path.exists(ANNOTATIONS_PATH):
        print("No annotations found. Run src/annotator.py to create ground truth.")
        sys.exit(1)

    annotations = load_annotations(ANNOTATIONS_PATH)
    print(f"Annotated files: {list(annotations.keys())}")

    run_ablation(ANNOTATIONS_PATH)
    run_accuracy_eval(ANNOTATIONS_PATH)
    run_robustness(ANNOTATIONS_PATH, n_runs=20)

    print("\nAll experiments complete. Results saved to results/")
