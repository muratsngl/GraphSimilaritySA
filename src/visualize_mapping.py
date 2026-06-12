"""
Physical-coordinate visualizer for the skeleton-correspondence pipeline.

Unlike visualize_graph.py (which lays graphs out topologically with
`breadthfirst` for intermediary paper figures), this viewer draws every graph
at its *real* world coordinates, so the figures are faithful to the actual
physical dimensions of the skeletons:

  Row 1   RAW  |  PRUNED        (target skeleton from the GLB, real coords)
  Row 2   IK RIG  |  TARGET     (the SA correspondence) + a mapping list

In Row 2 the best mapping is drawn by NUMBER: each IK joint shows its index,
and each target node shows the IK index assigned to it ('U' if unassigned).
The full mapping is also written top-to-bottom in a side panel.

Same stack as visualize_graph.py (Dash + dash_cytoscape); the only change is
`layout: 'preset'` with explicit per-node positions instead of `breadthfirst`.
"""

import sys
import os
import signal
import socket
import threading
import webbrowser
import time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pygltflib
import dash
from dash import html
import dash_cytoscape as cyto

from main import run_sa_pipeline, DEFAULT_GLB_PATH, SIGMA, LAMBDA_REPEL, GAMMA_PENALTY
from skeleton import h36m_rest_positions

GLB_PATH = DEFAULT_GLB_PATH

# Which two world axes form the drawing plane, and whether to flip vertically.
# Mixamo rigs are Y-up facing camera, so (x, y) is the front view; cytoscape's
# y grows downward, so we negate it to keep the character upright.
PROJ_AXES = (0, 1)
FLIP_Y = True


# --------------------------------------------------------------------------- #
# Forward kinematics: real world positions of every GLB node
# --------------------------------------------------------------------------- #

def _quat_to_matrix(q):
    """glTF quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz,       xz + wy],
        [xy + wz,       1 - (xx + zz), yz - wx],
        [xz - wy,       yz + wx,       1 - (xx + yy)],
    ])


def _local_matrix(node):
    """Local 4x4 transform of a glTF node (matrix, or composed from T/R/S)."""
    if node.matrix:
        # glTF stores matrices column-major.
        return np.array(node.matrix, dtype=float).reshape(4, 4, order='F')
    M = np.eye(4)
    if node.scale:
        M[0, 0], M[1, 1], M[2, 2] = node.scale
    if node.rotation:
        R = np.eye(4)
        R[:3, :3] = _quat_to_matrix(node.rotation)
        M = R @ M
    if node.translation:
        T = np.eye(4)
        T[:3, 3] = node.translation
        M = T @ M
    return M


def extract_world_positions(glb_path):
    """Return {node_index: (x, y, z)} world positions via forward kinematics."""
    gltf = pygltflib.GLTF2().load(glb_path)
    n = len(gltf.nodes)

    child_set = set()
    for node in gltf.nodes:
        if node.children:
            child_set.update(node.children)
    roots = [i for i in range(n) if i not in child_set]

    world = [None] * n
    stack = [(r, np.eye(4)) for r in roots]
    while stack:
        idx, parent_w = stack.pop()
        w = parent_w @ _local_matrix(gltf.nodes[idx])
        world[idx] = w
        for c in (gltf.nodes[idx].children or []):
            stack.append((c, w))

    return {i: world[i][:3, 3] for i in range(n) if world[i] is not None}


def project(pos3d):
    """Project a 3D world position onto the 2D drawing plane (screen coords)."""
    a, b = PROJ_AXES
    x, y = float(pos3d[a]), float(pos3d[b])
    return x, (-y if FLIP_Y else y)


def fit_positions(pos2d, box=500.0):
    """Rescale 2D positions (aspect-preserved, centred) so the larger span == box.

    The GLB graphs come in real world coordinates (~100 units) while the IK rig
    is ~1 unit. Cytoscape renders edge width as `width * zoom`, and `zoom` is the
    fit-to-panel factor, so a 100x larger graph gets zoomed 100x *out* and its
    edges shrink to hairlines. Normalizing every skeleton to the same model-space
    box equalizes the zoom, so all panels render edges at identical thickness.
    Proportions are preserved (single isotropic scale), so the skeletons keep
    their true shape.
    """
    xs = [p[0] for p in pos2d.values()]
    ys = [p[1] for p in pos2d.values()]
    if not xs:
        return dict(pos2d)
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    s = box / span
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    return {n: ((x - cx) * s, (y - cy) * s) for n, (x, y) in pos2d.items()}


# --------------------------------------------------------------------------- #
# Element builders
# --------------------------------------------------------------------------- #

def short_name(name):
    for p in ('mixamorig1', 'mixamorig'):
        if name.startswith(p):
            return name[len(p):] or name
    return name


def graph_elements(G, positions2d, labels, kinds=None):
    """Cytoscape elements with preset positions.

    positions2d : {node -> (x, y)} screen-space coords.
    labels      : {node -> str} node label.
    kinds       : optional {node -> str} class tag ('leaf'/'internal'/'unassigned')
                  used for styling.
    """
    elements = []
    for node in G.nodes():
        x, y = positions2d[node]
        data = {'id': str(node), 'label': labels.get(node, str(node))}
        if kinds is not None:
            data['kind'] = kinds.get(node, 'internal')
        elements.append({'data': data, 'position': {'x': x, 'y': y}})
    for u, v in G.edges():
        elements.append({'data': {'source': str(u), 'target': str(v)}})
    return elements


# --------------------------------------------------------------------------- #
# Stage everything
# --------------------------------------------------------------------------- #

print("Running pipeline + SA for visualization...")
try:
    world_pos = extract_world_positions(GLB_PATH)
    R = run_sa_pipeline(GLB_PATH, verbose=False)
except Exception as e:
    print(f"Pipeline failed: {e}")
    sys.exit(1)

raw_G, pruned_G = R['raw_G'], R['pruned_G']
G_ik, ik_ordered, trg_ordered = R['G_ik'], R['ik_ordered'], R['trg_ordered']
best_state = R['best_state']
ik_leaf_set = set(R['ik_leaves'].tolist())

print(f"Pipeline done — raw: {raw_G.number_of_nodes()} nodes, "
      f"pruned: {pruned_G.number_of_nodes()} nodes, "
      f"best energy: {R['best_energy']:.4f}")

# ---- Row 1: raw + pruned at physical coords (normalized to a common box) -- #
raw_pos = fit_positions({n: project(world_pos[n]) for n in raw_G.nodes() if n in world_pos})
pruned_pos = fit_positions({n: project(world_pos[n]) for n in pruned_G.nodes() if n in world_pos})
raw_labels = {n: short_name(d.get('name', str(n))) for n, d in raw_G.nodes(data=True)}
pruned_labels = {n: short_name(d.get('name', str(n))) for n, d in pruned_G.nodes(data=True)}

# ---- Row 2: the SA correspondence ----------------------------------------- #
# IK rig: drawn at canonical rest coords, labelled by joint index.
ik_rest = h36m_rest_positions()
ik_pos = fit_positions(
    {n: (lambda p: (p[0], -p[1] if FLIP_Y else p[1]))(ik_rest[n]) for n in G_ik.nodes()})
ik_labels = {ik_ordered[i]: str(i) for i in range(len(ik_ordered))}
ik_kinds = {ik_ordered[i]: ('leaf' if i in ik_leaf_set else 'internal')
            for i in range(len(ik_ordered))}

# Target: drawn at physical coords, labelled by the IK index assigned to it.
assigned = {int(trg_pos): ik_pos_i for ik_pos_i, trg_pos in enumerate(best_state)}
trg_pos2d = fit_positions({n: project(world_pos[n]) for n in pruned_G.nodes() if n in world_pos})
trg_labels, trg_kinds = {}, {}
for p, node_id in enumerate(trg_ordered):
    if p in assigned:
        ik_idx = assigned[p]
        trg_labels[node_id] = str(ik_idx)
        trg_kinds[node_id] = 'leaf' if ik_idx in ik_leaf_set else 'internal'
    else:
        trg_labels[node_id] = 'U'
        trg_kinds[node_id] = 'unassigned'

# ---- Mapping list (top to bottom) ----------------------------------------- #
mapping_rows = []
for i, trg_p in enumerate(best_state):
    ik_id = ik_ordered[i]
    trg_id = trg_ordered[int(trg_p)]
    ik_name = short_name(G_ik.nodes[ik_id].get('name', str(ik_id)))
    trg_name = short_name(pruned_G.nodes[trg_id].get('name', str(trg_id)))
    kind = 'leaf' if i in ik_leaf_set else 'internal'
    mapping_rows.append((i, ik_name, trg_name, kind))


# --------------------------------------------------------------------------- #
# Styling (dark theme, matching visualize_graph.py)
# --------------------------------------------------------------------------- #

STYLESHEET = [
    {'selector': 'node', 'style': {
        'label': 'data(label)',
        'color': '#0d1117',
        'background-color': '#58a6ff',
        'border-color': '#ffffff',
        'border-width': 2,
        'font-size': '15px',
        'font-weight': 'bold',
        'text-valign': 'center',
        'text-halign': 'center',
        'width': 32,
        'height': 32,
    }},
    {'selector': 'edge', 'style': {
        'line-color': '#ffffff',
        'target-arrow-color': '#ffffff',
        'target-arrow-shape': 'triangle',
        'arrow-scale': 1.0,
        'curve-style': 'bezier',
        'width': 4,
    }},
    {'selector': '[kind = "leaf"]', 'style': {'background-color': '#3fb950', 'border-color': '#ffffff'}},
    {'selector': '[kind = "unassigned"]', 'style': {
        'background-color': '#30363d', 'border-color': '#6e7681', 'color': '#8b949e'}},
]

PANEL_LABEL_STYLE = {
    'color': '#8b949e', 'fontSize': '12px', 'fontFamily': 'monospace',
    'padding': '8px 14px', 'borderBottom': '1px solid #21262d',
    'backgroundColor': '#161b22', 'flexShrink': 0,
}
PANEL_STYLE = {'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden'}
DIVIDER_STYLE = {'width': '1px', 'backgroundColor': '#21262d', 'flexShrink': 0}


def cyto_panel(graph_id, elements, label):
    return html.Div(style=PANEL_STYLE, children=[
        html.Div(label, style=PANEL_LABEL_STYLE),
        cyto.Cytoscape(
            id=graph_id,
            elements=elements,
            layout={'name': 'preset', 'fit': True, 'padding': 40},
            stylesheet=STYLESHEET,
            style={'width': '100%', 'height': '100%', 'backgroundColor': '#0d1117'},
        ),
    ])


def mapping_panel():
    header = html.Div("BEST MAPPING  (IK# · IK → Target)", style=PANEL_LABEL_STYLE)
    rows = []
    for i, ik_name, trg_name, kind in mapping_rows:
        color = '#3fb950' if kind == 'leaf' else '#58a6ff'
        rows.append(html.Div(
            style={'display': 'flex', 'gap': '8px', 'padding': '3px 14px',
                   'fontFamily': 'monospace', 'fontSize': '12px',
                   'borderBottom': '1px solid #161b22'},
            children=[
                html.Span(f"{i:>2}", style={'color': color, 'fontWeight': 'bold', 'width': '22px'}),
                html.Span(ik_name, style={'color': '#c9d1d9', 'width': '90px'}),
                html.Span("→", style={'color': '#6e7681'}),
                html.Span(trg_name, style={'color': '#8b949e'}),
            ]))
    return html.Div(
        style={'width': '300px', 'display': 'flex', 'flexDirection': 'column',
               'overflowY': 'auto', 'backgroundColor': '#0d1117',
               'borderLeft': '1px solid #21262d', 'flexShrink': 0},
        children=[header] + rows)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

app = dash.Dash(__name__)
app.title = "Skeleton Mapping Viewer (physical)"

ROW_STYLE = {'display': 'flex', 'flex': 1, 'overflow': 'hidden',
             'borderBottom': '1px solid #21262d'}

app.layout = html.Div(
    style={'backgroundColor': '#0d1117', 'height': '100vh', 'width': '100vw',
           'margin': 0, 'padding': 0, 'display': 'flex', 'flexDirection': 'column',
           'overflow': 'hidden'},
    children=[
        html.Div(
            style={'padding': '8px 20px', 'borderBottom': '1px solid #21262d',
                   'backgroundColor': '#161b22', 'flexShrink': 0,
                   'display': 'flex', 'alignItems': 'center', 'gap': '24px'},
            children=[
                html.Span("Skeleton Mapping Viewer · physical coordinates",
                          style={'color': '#58a6ff', 'fontSize': '13px',
                                 'fontFamily': 'monospace', 'fontWeight': 'bold'}),
                html.Span(f"σ={SIGMA}  λ={LAMBDA_REPEL}  γ={GAMMA_PENALTY}  ·  E={R['best_energy']:.4f}",
                          style={'color': '#8b949e', 'fontSize': '12px', 'fontFamily': 'monospace'}),
            ]),

        # Row 1: raw | pruned (target skeleton, real coords)
        html.Div(style=ROW_STYLE, children=[
            cyto_panel('raw-graph', graph_elements(raw_G, raw_pos, raw_labels),
                       f"RAW  ·  {raw_G.number_of_nodes()} nodes  ·  {raw_G.number_of_edges()} edges"),
            html.Div(style=DIVIDER_STYLE),
            cyto_panel('pruned-graph', graph_elements(pruned_G, pruned_pos, pruned_labels),
                       f"PRUNED  ·  {pruned_G.number_of_nodes()} nodes  ·  {pruned_G.number_of_edges()} edges"),
        ]),

        # Row 2: IK rig | target (assignment) | mapping list
        html.Div(style={'display': 'flex', 'flex': 1, 'overflow': 'hidden'}, children=[
            cyto_panel('ik-graph', graph_elements(G_ik, ik_pos, ik_labels, ik_kinds),
                       f"IK RIG  ·  {G_ik.number_of_nodes()} joints  ·  numbered"),
            html.Div(style=DIVIDER_STYLE),
            cyto_panel('target-graph', graph_elements(pruned_G, trg_pos2d, trg_labels, trg_kinds),
                       f"TARGET  ·  assigned IK#  ·  U = unassigned"),
            mapping_panel(),
        ]),
    ])


if __name__ == '__main__':
    with socket.socket() as _s:
        _s.bind(('', 0))
        port = _s.getsockname()[1]
    url = f'http://127.0.0.1:{port}'

    server_thread = threading.Thread(
        target=lambda: app.run(host='127.0.0.1', debug=False, use_reloader=False, port=port),
        daemon=True)
    server_thread.start()

    time.sleep(2)
    webbrowser.open(url)

    print(f"\nVisualization running at {url}")
    print("Press q + Enter to quit.\n")

    while True:
        try:
            if input().strip().lower() == 'q':
                break
        except (EOFError, KeyboardInterrupt):
            break

    os.kill(os.getpid(), signal.SIGINT)
