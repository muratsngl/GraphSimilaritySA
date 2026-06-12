"""
Utilities for loading annotations and matching them to the SA pipeline.

Usage example (accuracy evaluation):

    from annotation_utils import load_annotations, get_annotation_for_glb, match_annotation

    annotations  = load_annotations()
    entry        = get_annotation_for_glb(annotations, glb_path)   # None if not annotated
    if entry:
        gt_state = match_annotation(entry, pruned_G, trg_ordered)
        # gt_state[ik_pos] = position of the annotated bone in trg_ordered
        # -1 means that IK joint was not annotated or the bone name was not found
        annotated  = gt_state >= 0
        accuracy   = float(np.mean(best_state[annotated] == gt_state[annotated]))
        print(f"SA accuracy (annotated joints): {accuracy:.1%}")
"""

import os
import json
import numpy as np

_DEFAULT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'assets', 'annotations.json')
)


def load_annotations(path=None):
    """Load annotations.json and return the full dict.

    Returns an empty dict if the file does not exist yet.
    """
    if path is None:
        path = _DEFAULT_PATH
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def get_annotation_for_glb(annotations, glb_path):
    """Return the annotation entry for a GLB file, or None if not annotated.

    Parameters
    ----------
    annotations : dict
        Full dict returned by load_annotations().
    glb_path : str
        Full or relative path to the .glb file.  Only the basename is used
        as the lookup key, matching how the annotator stores entries.
    """
    fname = os.path.basename(glb_path)
    return annotations.get(fname)


def match_annotation(annotation_entry, pruned_G, trg_ordered):
    """Map an annotation entry to positional indices for accuracy measurement.

    Parameters
    ----------
    annotation_entry : dict
        One entry from annotations.json, e.g. the value of
        annotations['armature_01.glb'].  Must have an 'assignments' key
        of the form {str(ik_idx): bone_name}.
    pruned_G : networkx.DiGraph
        The pruned skeleton graph produced by the SA pipeline for the same GLB.
    trg_ordered : list[int]
        Ordered node-ID list used by the SA (same ordering as best_state).

    Returns
    -------
    np.ndarray, shape (17,), dtype int
        gt_state[ik_pos] = position of the annotated bone in trg_ordered,
        or -1 if that IK joint was not annotated or the bone name was not
        found in pruned_G (e.g. pruned away).

    Notes
    -----
    Comparison against best_state from run_sa_pipeline():

        annotated = gt_state >= 0
        accuracy  = float(np.mean(best_state[annotated] == gt_state[annotated]))

    Joints that were pruned or not annotated are excluded from the metric.
    """
    name_to_pos = {}
    for pos, node_id in enumerate(trg_ordered):
        name = pruned_G.nodes[node_id].get('name', '')
        if name:
            name_to_pos[name] = pos

    assignments = annotation_entry.get('assignments', {})
    gt_state = np.full(17, -1, dtype=int)

    for ik_idx_str, bone_name in assignments.items():
        ik_idx = int(ik_idx_str)
        if 0 <= ik_idx < 17 and bone_name in name_to_pos:
            gt_state[ik_idx] = name_to_pos[bone_name]

    return gt_state
