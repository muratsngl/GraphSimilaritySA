"""
Human3.6M 17-joint skeleton as a networkx DiGraph.
 
This is the standard 17-joint hierarchy used by deep-learning pose estimators
(VideoPose3D / Martinez et al. convention), and the SIK reference rig referred
to in the progress report. It is built to drop straight into the existing
pipeline (build_qap_ready_graph -> prune_leaves_iteratively -> Floyd-Warshall
-> normalize -> simulated_annealing): same DiGraph type, same per-node `name`
attribute, same per-edge `weight` attribute (parent -> child).
 
Joint indices (VideoPose3D ordering):
    0  Hip (root / pelvis)
    1  RHip
    2  RKnee
    3  RFoot       (right ankle / foot)
    4  LHip
    5  LKnee
    6  LFoot       (left ankle / foot)
    7  Spine
    8  Thorax
    9  Neck/Nose
    10 Head
    11 LShoulder
    12 LElbow
    13 LWrist
    14 RShoulder
    15 RElbow
    16 RWrist
 
Degree profile (undirected) for this tree:
    Hip (0)      : 3  (RHip, LHip, Spine)            -> internal (joint)
    Thorax (8)   : 3  (Neck, LShoulder, RShoulder)   -> internal (joint)
    leaves (deg 1): RFoot(3), LFoot(6), Head(10), LWrist(13), RWrist(16)
    everything else: deg 2 -> internal (regular chain node)
 
So the IK rig has 5 leaves and 12 internal nodes, exactly mirroring the
"all end-effectors are degree 1" property the report relies on.
"""
 
import networkx as nx
import numpy as np
 
 
# (parent, child) edges. Index == joint id; see module docstring.
H36M_NAMES = {
    0: "Hip",
    1: "RHip",
    2: "RKnee",
    3: "RFoot",
    4: "LHip",
    5: "LKnee",
    6: "LFoot",
    7: "Spine",
    8: "Thorax",
    9: "Neck",
    10: "Head",
    11: "LShoulder",
    12: "LElbow",
    13: "LWrist",
    14: "RShoulder",
    15: "RElbow",
    16: "RWrist",
}
 
# Parent of each joint (root has parent -1). VideoPose3D hierarchy.
H36M_PARENTS = {
    0: -1,
    1: 0,
    2: 1,
    3: 2,
    4: 0,
    5: 4,
    6: 5,
    7: 0,
    8: 7,
    9: 8,
    10: 9,
    11: 8,
    12: 11,
    13: 12,
    14: 8,
    15: 14,
    16: 15,
}
 
# Representative bone lengths (metres), used as default edge weights. These are
# rough adult-proportion values; override with real per-subject lengths when
# you have them. Keyed by child joint (the bone runs parent -> child).
H36M_BONE_LENGTHS = {
    1: 0.13,   # Hip -> RHip
    2: 0.44,   # RHip -> RKnee  (thigh)
    3: 0.45,   # RKnee -> RFoot (shin)
    4: 0.13,   # Hip -> LHip
    5: 0.44,   # LHip -> LKnee
    6: 0.45,   # LKnee -> LFoot
    7: 0.23,   # Hip -> Spine
    8: 0.23,   # Spine -> Thorax
    9: 0.12,   # Thorax -> Neck
    10: 0.11,  # Neck -> Head
    11: 0.15,  # Thorax -> LShoulder
    12: 0.28,  # LShoulder -> LElbow (upper arm)
    13: 0.25,  # LElbow -> LWrist    (forearm)
    14: 0.15,  # Thorax -> RShoulder
    15: 0.28,  # RShoulder -> RElbow
    16: 0.25,  # RElbow -> RWrist
}
 
 
def build_human36m_graph(bone_lengths=None):
    """Return the Human3.6M 17-joint skeleton as a networkx DiGraph.
 
    Matches the conventions produced by build_qap_ready_graph():
      - DiGraph with parent -> child edges
      - each node carries a `name` attribute
      - each edge carries a `weight` attribute (bone length)
 
    Parameters
    ----------
    bone_lengths : dict[int, float], optional
        Maps child joint id -> bone length. Defaults to H36M_BONE_LENGTHS.
        Pass your own (e.g. from a specific subject) to override.
    """
    if bone_lengths is None:
        bone_lengths = H36M_BONE_LENGTHS
 
    G = nx.DiGraph()
 
    for joint_id, name in H36M_NAMES.items():
        G.add_node(joint_id, name=name)
 
    for child, parent in H36M_PARENTS.items():
        if parent == -1:
            continue
        weight = float(bone_lengths.get(child, 0.0))
        G.add_edge(parent, child, weight=weight)
 
    print(f"Human3.6M skeleton built: "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G
 
 
def degree_classification(G):
    """Return (leaves, internals) as sorted lists of node ids, by undirected
    degree. Degree 1 == leaf (end-effector), degree > 1 == internal.
 
    Useful for feeding the SA optimizer's leaf/internal node arrays directly.
    """
    UG = G.to_undirected()
    leaves = sorted(n for n, d in UG.degree() if d == 1)
    internals = sorted(n for n, d in UG.degree() if d > 1)
    return leaves, internals
 
 
if __name__ == "__main__":
    G = build_human36m_graph()
 
    print("\nNodes:")
    for n in sorted(G.nodes()):
        print(f"  {n:2d}  {G.nodes[n]['name']}")
 
    print("\nEdges (parent -> child, weight):")
    for u, v, w in sorted(G.edges(data="weight")):
        print(f"  {u:2d} {G.nodes[u]['name']:<10s} -> "
              f"{v:2d} {G.nodes[v]['name']:<10s}  w={w:.3f}")
 
    leaves, internals = degree_classification(G)
    print(f"\nLeaves   ({len(leaves)}): {leaves}")
    print(f"  -> {[G.nodes[n]['name'] for n in leaves]}")
    print(f"Internals ({len(internals)}): {internals}")
    print(f"  -> {[G.nodes[n]['name'] for n in internals]}")
 
    # Sanity: tree with a single root, 16 edges, 17 nodes.
    roots = [n for n, d in G.in_degree() if d == 0]
    assert roots == [0], f"Expected single root [0], got {roots}"
    assert G.number_of_edges() == 16
    assert nx.is_weakly_connected(G)
    print("\nStructure valid: single root (Hip), connected, 16 bones.")