# Evaluation Specification
# SA Skeleton Correspondence — Coding Agent Instructions

Implement everything in one new file: `src/evaluate.py`.
Do not modify any existing file except to import from them.
Run from the project root with the virtualenv active.

---

## 0. Existing Code to Import

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from main import (run_sa_pipeline, build_ik_rig, build_hierarchy_constraints,
                  graph_to_normalized_matrix, SIGMA, LAMBDA_REPEL, GAMMA_PENALTY)
from simulated_annealing import build_affinity, energy
from skeleton import H36M_NAMES
import networkx as nx
import numpy as np
import csv, json, glob
```

---

## 1. Ground Truth Format — annotations.json

The user creates ground truth by running `src/annotator.py`, which saves
`assets/annotations.json`. Do NOT generate or hard-code this file.
Only read it.

The file structure is:

```json
{
  "Dozy.glb": {
    "description": "humanoid character",
    "assignments": {
      "0":  "mixamorig1:Hips",
      "1":  "mixamorig1:RightUpLeg",
      "2":  "mixamorig1:RightLeg",
      "3":  "mixamorig1:RightFoot",
      "4":  "mixamorig1:LeftUpLeg",
      "5":  "mixamorig1:LeftLeg",
      "6":  "mixamorig1:LeftFoot",
      "7":  "mixamorig1:Spine",
      "8":  "mixamorig1:Spine2",
      "9":  "mixamorig1:Neck",
      "10": "mixamorig1:Head",
      "11": "mixamorig1:LeftShoulder",
      "12": "mixamorig1:LeftForeArm",
      "13": "mixamorig1:LeftHand",
      "14": "mixamorig1:RightShoulder",
      "15": "mixamorig1:RightForeArm",
      "16": "mixamorig1:RightHand"
    }
  },
  "armature_01.glb": { ... }
}
```

Keys "0" through "16" are IK position indices. They correspond directly to the
sorted Human3.6M node IDs 0-16 (Hip=0, RHip=1, RKnee=2, ... RWrist=16).
Values are the full target bone names exactly as stored in the pruned graph's
`name` node attribute — same prefix, same casing, no stripping needed.

The annotator only shows pruned skeleton nodes, so every annotated bone name is
guaranteed to exist in the pruned graph. No ancestor fallback is needed.

---

## 2. Helper Functions to Implement

### 2a. load_annotations(annotations_path: str) -> dict
Reads `assets/annotations.json` and returns the parsed dict.
If the file does not exist, raise FileNotFoundError with a message saying:
"annotations.json not found. Run src/annotator.py first to create ground truth."

### 2b. build_ground_truth(glb_filename: str, pruned_G, trg_ordered: list,
                           annotations: dict) -> dict

Takes the filename of the GLB (basename only, e.g. "Dozy.glb"), the pruned
target graph, its ordered node list, and the full annotations dict.

Returns a dict mapping each IK position index (integer 0-16) to the position
index (integer) in trg_ordered of the correct target node.

Steps:
- Look up annotations[glb_filename]. If the key is missing, raise KeyError
  with the message: "No annotation found for {glb_filename}. Annotate it first."
- Get the "assignments" sub-dict: {ik_idx_str: target_bone_name}.
- For each ik_idx_str and target_bone_name:
  - Convert ik_idx_str to int to get ik_pos.
  - Search pruned_G.nodes(data=True) for a node whose "name" attribute equals
    target_bone_name exactly.
  - Get that node's position in trg_ordered (trg_ordered.index(node_id)).
  - Record {ik_pos: target_trg_position}.
- Return the completed dict.

If fewer than 17 entries exist in assignments (the annotation is incomplete),
print a warning: "WARNING: {glb_filename} has only {n}/17 joints annotated."
Still return whatever is available.

### 2c. hop_distance(pos_a: int, pos_b: int, pruned_G, trg_ordered: list) -> int
Computes the number of edges on the shortest undirected path between two nodes
in the pruned target graph.
- Convert pos_a and pos_b to node IDs using trg_ordered.
- Use nx.shortest_path_length on pruned_G.to_undirected().
- Return 0 if pos_a == pos_b.
- Return 999 if no path exists.

### 2d. detect_symmetry_flip(best_state, ik_ordered, pruned_G, trg_ordered) -> bool
Returns True if a left-right symmetry flip occurred.

Left IK joints are those whose H36M name starts with "L":
LHip(4), LKnee(5), LFoot(6), LShoulder(11), LElbow(12), LWrist(13) — 6 joints.
Get their positions in ik_ordered.

For each left IK joint at position i:
- Get best_state[i] to find the assigned target position.
- Get that target node's "name" attribute from pruned_G.
- Check if the name contains the substring "Right" (case-sensitive).
- If yes, count it as a flipped joint.

Return True if 4 or more of the 6 left joints landed on a "Right" target.

### 2e. compute_metrics(best_state, ik_ordered, pruned_G, trg_ordered,
                        ground_truth: dict, K_ik, K_trg, G_ik) -> dict

ground_truth is the dict from build_ground_truth:
{ik_pos_int: target_trg_pos_int} for all annotated joints.

SKIP any IK position not present in ground_truth (incomplete annotation).
Only compute metrics over the joints that have a ground truth entry.
Let n_annotated = number of annotated joints (should be 17 if fully annotated).

HOP DISTANCES:
For each ik_pos in ground_truth:
  gt_pos  = ground_truth[ik_pos]
  pred_pos = best_state[ik_pos]
  hop = hop_distance(pred_pos, gt_pos, pruned_G, trg_ordered)
Store as a list hop_list of length n_annotated.
Also store as a dict hop_by_joint: {H36M_NAMES[ik_ordered[ik_pos]]: hop}

EXACT ACCURACY (exact_acc):
  (count of hops == 0) / n_annotated * 100

NEAR ACCURACY (near_acc):
  (count of hops <= 1) / n_annotated * 100

WRONG LIMB RATE (wrong_limb):
  (count of hops >= 3) / n_annotated * 100

LIMB ACCURACY (limb_acc):
  Define 5 limbs using H36M joint names:
    LEFT_ARM   = ["LShoulder", "LElbow", "LWrist"]
    RIGHT_ARM  = ["RShoulder", "RElbow", "RWrist"]
    LEFT_LEG   = ["LHip", "LKnee", "LFoot"]
    RIGHT_LEG  = ["RHip", "RKnee", "RFoot"]
    SPINE_HEAD = ["Hip", "Spine", "Thorax", "Neck", "Head"]
  A limb is "correct" if ALL of its joints are in hop_by_joint AND
  all of their hops are <= 1.
  limb_acc = (number of correct limbs / 5) * 100.

AVR — ANCESTOR VIOLATION RATE (avr):
  Call build_hierarchy_constraints(G_ik, ik_ordered, pruned_G, trg_ordered)
  to get (ancestor_penalty_fn, _).
  Call ancestor_penalty_fn(best_state) to get the raw violation count.
  avr = (violation_count / 16) * 100.
  (16 = number of edges in Human3.6M.)

SYMMETRY FLIP (flip):
  Call detect_symmetry_flip(best_state, ik_ordered, pruned_G, trg_ordered).

QAP ENERGY — ATTRACTION ONLY (qap_energy):
  Call energy(best_state, K_ik, K_trg, lambda_repel=0.0).
  This isolates the structural quality score from penalty contributions.

Return:
{
    "exact_acc":    float,   # % joints at hop 0
    "near_acc":     float,   # % joints at hop <= 1
    "wrong_limb":   float,   # % joints at hop >= 3
    "limb_acc":     float,   # % of 5 limbs fully correct (all hops <= 1)
    "avr":          float,   # % of 16 IK edges with ancestor violation
    "flip":         bool,    # True = left-right symmetry flip detected
    "qap_energy":   float,   # raw QAP attraction term (lambda=0)
    "hop_list":     list,    # all hop distances, one per annotated joint
    "hop_by_joint": dict,    # {h36m_joint_name: hop_distance}
}

---

## 3. Experiment 1 — Ablation Study

### Purpose
Show what each component contributes. Run on every annotated GLB file.
Report mean metrics across all annotated files per configuration.

### The 5 configurations
All non-listed parameters stay fixed:
T=1.0, alpha=0.99, T_min=0.00001, iters_per_temp=100, seed=91

| Config name     | lambda_repel | gamma_penalty | n_restarts |
|-----------------|-------------|---------------|------------|
| Plain QAP       | 0.0         | 0.0           | 1          |
| Lambda only     | 0.5         | 0.0           | 1          |
| Gamma only      | 0.0         | 3.0           | 1          |
| Both penalties  | 0.5         | 3.0           | 1          |
| Full method     | 0.5         | 3.0           | 7          |

### Function signature
def run_ablation(annotations_path: str) -> list

Steps:
1. Load annotations from annotations_path.
2. For each annotated GLB filename:
   a. Build the full path: os.path.join(assets_dir, glb_filename).
   b. For each of the 5 configurations, call run_sa_pipeline with the config params.
   c. Rebuild K_ik and K_trg using build_affinity(D, sigma=0.2).
      D_ik and D_trg come from graph_to_normalized_matrix in the pipeline result.
   d. Build ground truth with build_ground_truth.
   e. Call compute_metrics and store the result.
3. For each configuration, average the metrics across all GLB files.
4. Return a list of 5 dicts, each with config name and averaged metrics.

### Output
Print a table with columns:
Config | Exact Acc% | Limb Acc% | AVR% | Flip Rate% | QAP Energy

Save to results/ablation.csv. Create results/ if missing.

---

## 4. Experiment 2 — Accuracy Evaluation

### Purpose
Main quantitative result. Full method on all annotated GLB files.

### Function signature
def run_accuracy_eval(annotations_path: str) -> dict

Steps:
1. Load annotations. Get the list of annotated filenames.
2. For each filename:
   a. Call run_sa_pipeline with all defaults (SIGMA, LAMBDA_REPEL, GAMMA_PENALTY,
      n_restarts=7, seed=91, T=1.0, alpha=0.99, T_min=0.00001, iters_per_temp=100).
   b. Rebuild K_ik and K_trg with build_affinity(D, sigma=0.2).
   c. Build ground truth with build_ground_truth.
   d. Call compute_metrics.
   e. Catch any exception, print a warning, and skip to next file.
3. Compute aggregates across all successful files:
   - mean and std of: exact_acc, near_acc, wrong_limb, limb_acc, avr, qap_energy
   - flip_rate = (count of flipped files / total files) * 100
   - flip_excluded_exact: mean exact_acc over non-flipped files only
   - flip_excluded_limb: mean limb_acc over non-flipped files only
4. Return the aggregation dict.

### Output
Print per-file table:
File | Exact Acc% | Limb Acc% | Wrong Limb% | AVR% | Flip | QAP Energy

Then print:
=== AGGREGATE (N files) ===
Exact accuracy:           mean +/- std %
Near accuracy (hop<=1):   mean +/- std %
Wrong limb rate:          mean +/- std %
Limb accuracy:            mean +/- std %
AVR:                      mean +/- std %
Flip rate:                X%
Flip-excluded exact acc:  X%
Flip-excluded limb acc:   X%

Save per-file rows to results/accuracy.csv.
Save aggregate to results/accuracy_summary.csv as a single row.

---

## 5. Experiment 3 — Robustness

### Purpose
Show SA is stable across seeds. Use the first annotated GLB file as the test rig.

### Function signature
def run_robustness(annotations_path: str, n_runs: int = 20) -> dict

Steps:
1. Load annotations. Use the first key as the test GLB filename.
2. Build the full path: os.path.join(assets_dir, glb_filename).
3. Build ground truth once (it does not change across runs).
4. Run the full pipeline n_runs times with seeds 0 through n_runs-1.
   All other params at full defaults (same as Experiment 2).
5. For each run, call compute_metrics and store: exact_acc, limb_acc,
   qap_energy, flip, and the full history list.
6. Compute: mean and std of qap_energy, exact_acc, limb_acc.
   Count flips.
7. Save the history of the best run (lowest qap_energy) as the convergence data.

### Output
Print per-run table:
Run | Seed | Exact Acc% | Limb Acc% | QAP Energy | Flip

Then print:
=== ROBUSTNESS SUMMARY (N runs) ===
QAP energy:     mean +/- std
Exact accuracy: mean +/- std %
Limb accuracy:  mean +/- std %
Flip rate:      X / N runs

Save per-run rows to results/robustness.csv.
Save best run's energy history to results/convergence.csv
with columns: step, energy. One row per entry, step starts at 0.

---

## 6. Main Entry Point

if __name__ == "__main__":
    ANNOTATIONS_PATH = os.path.join(
        os.path.dirname(__file__), '..', 'assets', 'annotations.json')
    ASSETS_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'assets'))

    1. Check that annotations.json exists. If not, print:
       "No annotations found. Run src/annotator.py to create ground truth."
       and exit.

    2. Load annotations and print the list of annotated files.

    3. Run all three experiments in order:
       run_ablation(ANNOTATIONS_PATH)
       run_accuracy_eval(ANNOTATIONS_PATH)
       run_robustness(ANNOTATIONS_PATH, n_runs=20)

    4. Print: "All experiments complete. Results saved to results/"

---

## 7. Output Files

| File                         | Rows                   | Used for                   |
|------------------------------|------------------------|----------------------------|
| results/ablation.csv         | 5 (one per config)     | Ablation table in paper    |
| results/accuracy.csv         | N (one per GLB file)   | Per-character accuracy     |
| results/accuracy_summary.csv | 1 (aggregate)          | Headline numbers in paper  |
| results/robustness.csv       | 20 (one per run)       | Robustness table           |
| results/convergence.csv      | M (one per SA step)    | Convergence curve figure   |

---

## 8. Parameter Reference

| Parameter          | Value    | Source                      |
|--------------------|----------|-----------------------------|
| sigma              | 0.2      | main.py SIGMA               |
| lambda_repel       | 0.5      | main.py LAMBDA_REPEL        |
| gamma_penalty      | 3.0      | main.py GAMMA_PENALTY       |
| n_restarts         | 7        | main.py run_sa_pipeline     |
| seed               | 91       | main.py run_sa_pipeline     |
| T                  | 1.0      | main.py run_sa_pipeline     |
| alpha              | 0.99     | main.py run_sa_pipeline     |
| T_min              | 0.00001  | main.py run_sa_pipeline     |
| iters_per_temp     | 100      | main.py run_sa_pipeline     |
| Robustness n_runs  | 20       | evaluate.py                 |
| Flip threshold     | 4 of 6   | detect_symmetry_flip        |
| Limb correct       | hop <= 1 | compute_metrics             |
| Wrong limb         | hop >= 3 | compute_metrics             |

---

## 9. Dependencies

No new packages. Uses only:
- networkx, numpy  (already in requirements.txt)
- csv, json, os, glob  (Python standard library)
