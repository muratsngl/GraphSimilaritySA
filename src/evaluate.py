"""
SA Skeleton Correspondence — Evaluation Suite.

Two experiments:
  1. Eval      — ablation table (6 configs × all annotated GLBs) +
                 per-file detail for the full method
  2. Robustness — stability across 20 seeds on the first annotated GLB

Run:  python src/evaluate.py
Results saved to results/results.csv and results/robustness.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import csv
import json
import numpy as np
import networkx as nx

from main import (run_sa_pipeline, SIGMA, LAMBDA_REPEL, GAMMA_PENALTY, LAMBDA_LATERAL)
from simulated_annealing import energy
from skeleton import H36M_NAMES

# ── paths ─────────────────────────────────────────────────────────────────────

ASSETS_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results'))

# ── logger ────────────────────────────────────────────────────────────────────

_log_fh = None


def log(msg=''):
    print(msg)
    if _log_fh is not None:
        _log_fh.write(str(msg) + '\n')
        _log_fh.flush()


# ── ablation configs ──────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    {'name': 'Plain QAP',      'lambda_repel': 0.0, 'gamma_penalty': 0.0, 'lambda_lateral': 0.0, 'n_restarts': 1},
    {'name': 'Lambda only',    'lambda_repel': 0.5, 'gamma_penalty': 0.0, 'lambda_lateral': 0.0, 'n_restarts': 1},
    {'name': 'Gamma only',     'lambda_repel': 0.0, 'gamma_penalty': 3.0, 'lambda_lateral': 0.0, 'n_restarts': 1},
    {'name': 'Both penalties', 'lambda_repel': 0.5, 'gamma_penalty': 3.0, 'lambda_lateral': 0.0, 'n_restarts': 1},
    {'name': 'Lateral added',  'lambda_repel': 0.5, 'gamma_penalty': 3.0, 'lambda_lateral': 1.0, 'n_restarts': 1},
    {'name': 'Full method',    'lambda_repel': 0.5, 'gamma_penalty': 3.0, 'lambda_lateral': 1.0, 'n_restarts': 7,
     'show_per_file': True},
]

# ── limb definitions ──────────────────────────────────────────────────────────

LIMBS = {
    'LEFT_ARM':   ['LShoulder', 'LElbow', 'LWrist'],
    'RIGHT_ARM':  ['RShoulder', 'RElbow', 'RWrist'],
    'LEFT_LEG':   ['LHip', 'LKnee', 'LFoot'],
    'RIGHT_LEG':  ['RHip', 'RKnee', 'RFoot'],
    'SPINE_HEAD': ['Hip', 'Spine', 'Thorax', 'Neck', 'Head'],
}

# ── annotation helpers ────────────────────────────────────────────────────────

def load_annotations(annotations_path: str) -> dict:
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(
            "annotations.json not found. Run src/annotator.py first.")
    with open(annotations_path) as f:
        return json.load(f)


def build_ground_truth(glb_filename: str, pruned_G, trg_ordered: list,
                       annotations: dict) -> dict:
    """Return {ik_pos: trg_position} for all annotated joints."""
    if glb_filename not in annotations:
        raise KeyError(f"No annotation for {glb_filename}.")

    assignments = annotations[glb_filename].get('assignments', {})
    if len(assignments) < 17:
        log(f"WARNING: {glb_filename} only {len(assignments)}/17 joints annotated.")

    name_to_node = {d.get('name', ''): nid for nid, d in pruned_G.nodes(data=True)}
    ground_truth = {}
    for ik_idx_str, bone_name in assignments.items():
        node_id = name_to_node.get(bone_name)
        if node_id is None:
            log(f"WARNING: bone '{bone_name}' not in pruned graph — skipping.")
            continue
        ground_truth[int(ik_idx_str)] = trg_ordered.index(node_id)
    return ground_truth

# ── flip detection ────────────────────────────────────────────────────────────

def detect_symmetry_flip(best_state, lat_ik, lat_trg, lat_threshold=0.1) -> bool:
    """True if >= 50% of clearly-lateral IK joints land on the wrong side.

    Checks the sign of the x-component of each joint's laterality unit vector.
    Works for any rig regardless of bone naming conventions.
    """
    mismatches = total = 0
    for i, trg_pos in enumerate(best_state):
        ik_x  = lat_ik[i][0]
        trg_x = lat_trg[int(trg_pos)][0]
        if abs(ik_x) > lat_threshold and abs(trg_x) > lat_threshold:
            total += 1
            if ik_x * trg_x < 0:
                mismatches += 1
    return total > 0 and mismatches / total >= 0.5

# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(best_state, ik_ordered, pruned_G, trg_ordered,
                    ground_truth: dict, lat_ik, lat_trg) -> dict:
    """Compute exact_acc, limb_acc, flip for one SA result vs ground truth."""
    UG = pruned_G.to_undirected()
    n  = len(ground_truth)

    hop_by_joint = {}
    for ik_pos, gt_pos in ground_truth.items():
        pred = int(best_state[ik_pos])
        if pred == gt_pos:
            hop = 0
        else:
            try:
                hop = nx.shortest_path_length(UG, trg_ordered[pred], trg_ordered[gt_pos])
            except nx.NetworkXNoPath:
                hop = 999
        hop_by_joint[H36M_NAMES[ik_ordered[ik_pos]]] = hop

    exact_acc = sum(1 for h in hop_by_joint.values() if h == 0) / n * 100
    limb_acc  = sum(
        1 for joints in LIMBS.values()
        if all(hop_by_joint.get(j, 999) <= 1 for j in joints)
    ) / len(LIMBS) * 100
    flip = detect_symmetry_flip(best_state, lat_ik, lat_trg)

    return {'exact_acc': exact_acc, 'limb_acc': limb_acc, 'flip': flip}

# ── Experiment 1 — Eval ───────────────────────────────────────────────────────

def run_eval(annotations_path: str,
             assets_dir: str = ASSETS_DIR,
             sigma: float = SIGMA,
             seed: int = 91) -> list:
    """Ablation table across all configs + per-file breakdown for the full method."""
    annotations = load_annotations(annotations_path)
    glb_files   = list(annotations.keys())
    log(f"\n=== EVAL ({len(glb_files)} file(s), sigma={sigma}) ===")

    summary_rows    = []
    full_per_file   = []

    for cfg in ABLATION_CONFIGS:
        cfg_metrics = []
        log(f"\n  Config: {cfg['name']}")

        for glb_filename in glb_files:
            glb_path = os.path.join(assets_dir, glb_filename)
            try:
                R = run_sa_pipeline(
                    glb_path=glb_path,
                    seed=seed,
                    sigma=sigma,
                    lambda_repel=cfg['lambda_repel'],
                    gamma_penalty=cfg['gamma_penalty'],
                    lambda_lateral=cfg['lambda_lateral'],
                    n_restarts=cfg['n_restarts'],
                    verbose=False,
                )
                gt = build_ground_truth(
                    glb_filename, R['pruned_G'], R['trg_ordered'], annotations)
                m = compute_metrics(
                    R['best_state'], R['ik_ordered'], R['pruned_G'], R['trg_ordered'],
                    gt, R['lat_ik'], R['lat_trg'])
                cfg_metrics.append(m)
                log(f"    {glb_filename}: exact={m['exact_acc']:.1f}%  limb={m['limb_acc']:.1f}%")

                if cfg.get('show_per_file'):
                    full_per_file.append({**m, 'file': glb_filename})

            except Exception as e:
                log(f"    WARNING: {glb_filename} skipped — {e}")

        if not cfg_metrics:
            continue

        summary_rows.append({
            'config':    cfg['name'],
            'exact_acc': float(np.mean([m['exact_acc'] for m in cfg_metrics])),
            'limb_acc':  float(np.mean([m['limb_acc']  for m in cfg_metrics])),
            'flip_rate': float(np.mean([float(m['flip']) for m in cfg_metrics])) * 100,
        })

    # ── ablation summary table ────────────────────────────────────────────────
    log(f"\n{'Config':<18} {'Exact%':>8} {'Limb%':>8} {'Flip%':>8}")
    log("-" * 46)
    for r in summary_rows:
        log(f"{r['config']:<18} {r['exact_acc']:>8.1f} {r['limb_acc']:>8.1f} {r['flip_rate']:>8.1f}")

    # ── full method per-file detail ───────────────────────────────────────────
    if full_per_file:
        N = len(full_per_file)
        log(f"\n=== FULL METHOD — per file ({N}) ===")
        log(f"\n{'File':<36} {'Exact%':>7} {'Limb%':>7} {'Flip':>5}")
        log("-" * 58)
        for m in full_per_file:
            log(f"{m['file']:<36} {m['exact_acc']:>7.1f} {m['limb_acc']:>7.1f} "
                f"{'YES' if m['flip'] else 'no':>5}")

        exact_vals = [m['exact_acc'] for m in full_per_file]
        limb_vals  = [m['limb_acc']  for m in full_per_file]
        flip_rate  = np.mean([float(m['flip']) for m in full_per_file]) * 100
        log(f"\nExact:  {np.mean(exact_vals):.1f} ± {np.std(exact_vals):.1f} %")
        log(f"Limb:   {np.mean(limb_vals):.1f} ± {np.std(limb_vals):.1f} %")
        log(f"Flip:   {flip_rate:.1f} %")

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'results.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['config', 'exact_acc', 'limb_acc', 'flip_rate'])
        writer.writeheader()
        writer.writerows(summary_rows)
    log(f"\nSaved {path}")

    return summary_rows

# ── Experiment 2 — Robustness ─────────────────────────────────────────────────

def run_robustness(annotations_path: str,
                   n_runs: int = 20,
                   assets_dir: str = ASSETS_DIR,
                   sigma: float = SIGMA,
                   lambda_repel: float = LAMBDA_REPEL,
                   gamma_penalty: float = GAMMA_PENALTY,
                   lambda_lateral: float = LAMBDA_LATERAL,
                   n_restarts: int = 7) -> dict:
    """Stability test: full method run n_runs times (seeds 0..n_runs-1)."""
    annotations  = load_annotations(annotations_path)
    glb_filename = next(iter(annotations))
    glb_path     = os.path.join(assets_dir, glb_filename)
    log(f"\n=== ROBUSTNESS ({n_runs} runs, file={glb_filename}) ===")

    R0 = run_sa_pipeline(glb_path=glb_path, seed=0, sigma=sigma,
                         lambda_repel=lambda_repel, gamma_penalty=gamma_penalty,
                         lambda_lateral=lambda_lateral, n_restarts=n_restarts,
                         verbose=False)
    ground_truth = build_ground_truth(
        glb_filename, R0['pruned_G'], R0['trg_ordered'], annotations)
    K_ik  = R0['K_ik']
    K_trg = R0['K_trg']

    per_run          = []
    best_energy_seen = np.inf
    best_history     = None

    log(f"\n{'Run':>4} {'Seed':>5} {'Exact%':>8} {'Limb%':>8} {'QAP E':>10} {'Flip':>5}")
    log("-" * 44)

    for run_idx in range(n_runs):
        try:
            R = run_sa_pipeline(glb_path=glb_path, seed=run_idx, sigma=sigma,
                                lambda_repel=lambda_repel, gamma_penalty=gamma_penalty,
                                lambda_lateral=lambda_lateral, n_restarts=n_restarts,
                                verbose=False)
            m = compute_metrics(
                R['best_state'], R['ik_ordered'], R['pruned_G'], R['trg_ordered'],
                ground_truth, R['lat_ik'], R['lat_trg'])
            qap_e = energy(R['best_state'], K_ik, K_trg, lambda_repel=0.0)

            if R['best_energy'] < best_energy_seen:
                best_energy_seen = R['best_energy']
                best_history     = R['history']

            row = {'run': run_idx, 'seed': run_idx,
                   'exact_acc': m['exact_acc'], 'limb_acc': m['limb_acc'],
                   'qap_energy': qap_e, 'flip': m['flip']}
            per_run.append(row)
            log(f"{run_idx:>4} {run_idx:>5} {m['exact_acc']:>8.1f} {m['limb_acc']:>8.1f} "
                f"{qap_e:>10.4f} {'YES' if m['flip'] else 'no':>5}")

        except Exception as e:
            log(f"  Run {run_idx} failed — {e}")

    if not per_run:
        log("No runs completed.")
        return {}

    N = len(per_run)
    summary = {
        'n_runs':          N,
        'exact_acc_mean':  float(np.mean([r['exact_acc']  for r in per_run])),
        'exact_acc_std':   float(np.std( [r['exact_acc']  for r in per_run])),
        'limb_acc_mean':   float(np.mean([r['limb_acc']   for r in per_run])),
        'limb_acc_std':    float(np.std( [r['limb_acc']   for r in per_run])),
        'qap_energy_mean': float(np.mean([r['qap_energy'] for r in per_run])),
        'qap_energy_std':  float(np.std( [r['qap_energy'] for r in per_run])),
        'n_flips':         sum(1 for r in per_run if r['flip']),
    }

    log(f"\n=== ROBUSTNESS SUMMARY ({N} runs) ===")
    log(f"Exact:      {summary['exact_acc_mean']:.1f} ± {summary['exact_acc_std']:.1f} %")
    log(f"Limb:       {summary['limb_acc_mean']:.1f} ± {summary['limb_acc_std']:.1f} %")
    log(f"QAP energy: {summary['qap_energy_mean']:.4f} ± {summary['qap_energy_std']:.4f}")
    log(f"Flip rate:  {summary['n_flips']} / {N} runs")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rob_path = os.path.join(RESULTS_DIR, 'robustness.csv')
    with open(rob_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'seed', 'exact_acc',
                                               'limb_acc', 'qap_energy', 'flip'])
        writer.writeheader()
        writer.writerows(per_run)
    log(f"Saved {rob_path}")

    if best_history is not None:
        conv_path = os.path.join(RESULTS_DIR, 'convergence.csv')
        with open(conv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'energy'])
            writer.writeheader()
            writer.writerows({'step': i, 'energy': e}
                             for i, e in enumerate(best_history))
        log(f"Saved {conv_path}")

    return summary

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import datetime

    ANNOTATIONS_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'assets', 'annotations.json')

    if not os.path.exists(ANNOTATIONS_PATH):
        print("No annotations found. Run src/annotator.py to create ground truth.")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    _log_path = os.path.join(RESULTS_DIR, 'results.txt')

    _log_fh = open(_log_path, 'w')
    try:
        log(f"Evaluation run — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Annotated files: {list(load_annotations(ANNOTATIONS_PATH).keys())}")

        run_eval(ANNOTATIONS_PATH)
        run_robustness(ANNOTATIONS_PATH, n_runs=20)

        log(f"\nAll experiments complete. Results in {RESULTS_DIR}/")
    finally:
        _log_fh.close()
