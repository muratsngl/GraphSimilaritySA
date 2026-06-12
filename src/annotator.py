"""
Skeleton Annotator — manual bone assignment tool.

Scans assets/ for all .glb files, renders each as a pruned skeleton graph
alongside the H36M IK rig (joints 0-16), and lets you click-to-assign
correspondences. Results are saved to assets/annotations.json.

Interaction:
  1. Pick a skeleton from the dropdown.
  2. Click an IK joint on the left panel (turns orange = selected source).
  3. Click a target bone on the right panel (turns green = assigned).
  4. Repeat for all 17 joints, fill in a description, then Save.
  Switching skeletons auto-loads any existing annotation for that file.

Run:  python src/annotator.py
URL:  http://127.0.0.1:8052
Quit: q + Enter
"""

import sys
import os
import json
import signal
import threading
import webbrowser
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pygltflib
import dash
from dash import html, dcc, Input, Output, State, callback_context, no_update
import dash_cytoscape as cyto

from skeleton_extraction import build_qap_ready_graph
from graph_modification import prune_leaves_iteratively
from skeleton import build_human36m_graph, h36m_rest_positions, H36M_NAMES

# ── paths ─────────────────────────────────────────────────────────────────────

ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
ANNOTATIONS_PATH = os.path.join(ASSETS_DIR, 'annotations.json')


# ── forward-kinematics helpers (inline; visualize_mapping.py runs SA on import) ──

def _quat_to_matrix(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    return np.array([
        [1-(yy+zz), xy-wz,     xz+wy],
        [xy+wz,     1-(xx+zz), yz-wx],
        [xz-wy,     yz+wx,     1-(xx+yy)],
    ])


def _local_matrix(node):
    if node.matrix:
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
        idx, pw = stack.pop()
        w = pw @ _local_matrix(gltf.nodes[idx])
        world[idx] = w
        for c in (gltf.nodes[idx].children or []):
            stack.append((c, w))
    return {i: world[i][:3, 3] for i in range(n) if world[i] is not None}


def project(pos3d):
    return float(pos3d[0]), -float(pos3d[1])


def fit_positions(pos2d, box=500.0):
    if not pos2d:
        return {}
    xs = [p[0] for p in pos2d.values()]
    ys = [p[1] for p in pos2d.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny) or 1.0
    s = box / span
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    return {n: ((x - cx) * s, (y - cy) * s) for n, (x, y) in pos2d.items()}


def short_name(name):
    for prefix in ('mixamorig1', 'mixamorig'):
        if name.startswith(prefix):
            return name[len(prefix):] or name
    return name


# ── annotation I/O ────────────────────────────────────────────────────────────

def load_annotations():
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH) as f:
            return json.load(f)
    return {}


def save_annotations(data):
    with open(ANNOTATIONS_PATH, 'w') as f:
        json.dump(data, f, indent=2)


# ── startup: scan assets and build all skeleton graphs ───────────────────────

def scan_glb_files():
    return sorted(f for f in os.listdir(ASSETS_DIR) if f.lower().endswith('.glb'))


def build_skeleton_data(glb_path):
    raw_G = build_qap_ready_graph(glb_path)
    pruned_G, _ = prune_leaves_iteratively(raw_G)
    world_pos = extract_world_positions(glb_path)
    return pruned_G, world_pos


print("Scanning assets/ for GLB files...")
GLB_FILES = scan_glb_files()
if not GLB_FILES:
    print("No .glb files found in assets/. Exiting.")
    sys.exit(1)
print(f"Found: {GLB_FILES}")

SKELETON_DATA = {}  # filename → (pruned_G, world_positions)
for _fname in GLB_FILES:
    _fpath = os.path.join(ASSETS_DIR, _fname)
    print(f"  Building graph for {_fname}...")
    SKELETON_DATA[_fname] = build_skeleton_data(_fpath)

# IK rig — built once, never changes
G_IK = build_human36m_graph()
IK_ORDERED = sorted(G_IK.nodes())          # [0, 1, ..., 16]
IK_IDX_OF = {n: i for i, n in enumerate(IK_ORDERED)}
N_IK = len(IK_ORDERED)

_ik_rest = h36m_rest_positions()
IK_POS = fit_positions({n: (p[0], -p[1]) for n, p in _ik_rest.items()})

INITIAL_ANNOTATIONS = load_annotations()


# ── session helpers: convert between node-ID (cytoscape) and name (JSON) ─────

def _name_to_node_id(pruned_G, name):
    for node_id, data in pruned_G.nodes(data=True):
        if data.get('name') == name:
            return str(node_id)
    return None


def assignments_to_names(skeleton_name, assignments):
    """Session {str(ik_idx): str(node_id)} → JSON {str(ik_idx): bone_name}."""
    pruned_G, _ = SKELETON_DATA[skeleton_name]
    result = {}
    for ik_idx_str, node_id_str in assignments.items():
        node_id = int(node_id_str)
        if pruned_G.has_node(node_id):
            result[ik_idx_str] = pruned_G.nodes[node_id].get('name', node_id_str)
    return result


def names_to_assignments(skeleton_name, name_assignments):
    """JSON {str(ik_idx): bone_name} → session {str(ik_idx): str(node_id)}."""
    pruned_G, _ = SKELETON_DATA[skeleton_name]
    result = {}
    for ik_idx_str, name in name_assignments.items():
        node_id_str = _name_to_node_id(pruned_G, name)
        if node_id_str is not None:
            result[ik_idx_str] = node_id_str
    return result


# ── cytoscape element builders ────────────────────────────────────────────────

def build_ik_elements(selected_ik_id=None, assigned_ik_node_ids=None):
    """IK rig elements. selected_ik_id and assigned_ik_node_ids are str node ids."""
    if assigned_ik_node_ids is None:
        assigned_ik_node_ids = set()
    elements = []
    for node_id in G_IK.nodes():
        sid = str(node_id)
        ik_idx = IK_IDX_OF[node_id]
        name = H36M_NAMES.get(node_id, sid)
        label = f"{ik_idx}:{name}"
        if sid == selected_ik_id:
            kind = 'selected'
        elif sid in assigned_ik_node_ids:
            kind = 'assigned'
        else:
            kind = 'default'
        x, y = IK_POS[node_id]
        elements.append({
            'data': {'id': sid, 'label': label, 'kind': kind},
            'position': {'x': x, 'y': y},
        })
    for u, v in G_IK.edges():
        elements.append({'data': {'source': str(u), 'target': str(v)}})
    return elements


def build_target_elements(skeleton_name, assignments=None):
    """Target skeleton elements. assignments: {str(ik_idx): str(node_id)}."""
    if assignments is None:
        assignments = {}
    if skeleton_name not in SKELETON_DATA:
        return [], {'name': 'breadthfirst', 'directed': True, 'padding': 40}

    pruned_G, world_pos = SKELETON_DATA[skeleton_name]
    assigned_target_ids = {v: k for k, v in assignments.items()}  # node_id_str → ik_idx_str

    raw_pos = {n: project(world_pos[n]) for n in pruned_G.nodes() if n in world_pos}
    use_preset = bool(raw_pos)

    if use_preset:
        pos2d = fit_positions(raw_pos)
        # Nodes missing from world_pos land at centroid — rare, non-blocking
        if len(raw_pos) < pruned_G.number_of_nodes():
            cx = sum(x for x, _ in pos2d.values()) / len(pos2d)
            cy = sum(y for _, y in pos2d.values()) / len(pos2d)
            for n in pruned_G.nodes():
                if n not in pos2d:
                    pos2d[n] = (cx, cy)

    roots = [str(n) for n in pruned_G.nodes() if pruned_G.in_degree(n) == 0]
    layout = (
        {'name': 'preset', 'fit': True, 'padding': 40}
        if use_preset else
        {'name': 'breadthfirst', 'directed': True, 'roots': roots,
         'spacingFactor': 1.75, 'animate': False, 'padding': 40}
    )

    elements = []
    for node_id, data in pruned_G.nodes(data=True):
        sid = str(node_id)
        raw_node_name = data.get('name', sid)
        if sid in assigned_target_ids:
            ik_idx_str = assigned_target_ids[sid]
            ik_node = IK_ORDERED[int(ik_idx_str)]
            ik_name = H36M_NAMES.get(ik_node, ik_idx_str)
            label = f"{ik_idx_str}:{ik_name}"
            kind = 'assigned'
        else:
            label = short_name(raw_node_name)
            kind = 'default'
        elem = {'data': {'id': sid, 'label': label, 'node_name': raw_node_name, 'kind': kind}}
        if use_preset:
            elem['position'] = {'x': pos2d[node_id][0], 'y': pos2d[node_id][1]}
        elements.append(elem)

    for u, v in pruned_G.edges():
        elements.append({'data': {'source': str(u), 'target': str(v)}})

    return elements, layout


# ── assignment table ──────────────────────────────────────────────────────────

def make_assignment_table(assignments, skeleton_name):
    rows = []
    pruned_G = SKELETON_DATA[skeleton_name][0] if skeleton_name in SKELETON_DATA else None
    for i in range(N_IK):
        ik_name = H36M_NAMES.get(IK_ORDERED[i], str(i))
        ik_idx_str = str(i)
        if ik_idx_str in assignments and pruned_G is not None:
            node_id = int(assignments[ik_idx_str])
            if pruned_G.has_node(node_id):
                target_label = short_name(pruned_G.nodes[node_id].get('name', str(node_id)))
            else:
                target_label = '?'
            color = '#3fb950'
        else:
            target_label = '—'
            color = '#6e7681'
        rows.append(html.Div(
            style={
                'display': 'flex', 'gap': '8px', 'padding': '2px 10px',
                'fontFamily': 'monospace', 'fontSize': '11px',
                'borderBottom': '1px solid #161b22',
            },
            children=[
                html.Span(f"{i:>2}", style={'color': color, 'fontWeight': 'bold',
                                            'width': '18px', 'flexShrink': 0}),
                html.Span(ik_name, style={'color': '#c9d1d9', 'width': '88px',
                                          'flexShrink': 0, 'overflow': 'hidden'}),
                html.Span("→", style={'color': '#6e7681'}),
                html.Span(target_label, style={'color': '#8b949e', 'overflow': 'hidden',
                                               'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'}),
            ],
        ))
    return rows


# ── Dash app ──────────────────────────────────────────────────────────────────

STYLESHEET = [
    {'selector': 'node', 'style': {
        'label': 'data(label)',
        'color': '#c9d1d9',
        'background-color': '#30363d',
        'border-color': '#6e7681',
        'border-width': 2,
        'font-size': '10px',
        'text-valign': 'center',
        'text-halign': 'right',
        'text-margin-x': 8,
        'text-background-color': '#0d1117',
        'text-background-opacity': 0.7,
        'text-background-padding': '2px',
        'width': 24,
        'height': 24,
    }},
    {'selector': 'edge', 'style': {
        'line-color': '#3d444d',
        'target-arrow-color': '#3d444d',
        'target-arrow-shape': 'triangle',
        'arrow-scale': 0.7,
        'curve-style': 'bezier',
        'width': 2,
    }},
    {'selector': '[kind = "selected"]', 'style': {
        'background-color': '#f0883e',
        'border-color': '#ffa657',
        'border-width': 3,
        'color': '#fff8',
        'z-index': 10,
    }},
    {'selector': '[kind = "assigned"]', 'style': {
        'background-color': '#238636',
        'border-color': '#3fb950',
        'border-width': 2,
        'color': '#fff',
    }},
]

_S = {'fontFamily': 'monospace', 'fontSize': '12px'}
HEADER_STYLE = {
    'padding': '8px 16px', 'borderBottom': '1px solid #21262d',
    'backgroundColor': '#161b22', 'flexShrink': 0,
    'display': 'flex', 'alignItems': 'center', 'gap': '12px',
}
PANEL_LABEL_STYLE = {
    'color': '#8b949e', 'fontSize': '11px', 'fontFamily': 'monospace',
    'padding': '6px 14px', 'borderBottom': '1px solid #21262d',
    'backgroundColor': '#161b22', 'flexShrink': 0,
}
PANEL_STYLE = {'flex': 1, 'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden'}
DIVIDER_STYLE = {'width': '1px', 'backgroundColor': '#21262d', 'flexShrink': 0}
BTN = {**_S, 'backgroundColor': '#21262d', 'color': '#c9d1d9',
       'border': '1px solid #30363d', 'borderRadius': '6px',
       'padding': '4px 12px', 'cursor': 'pointer'}
BTN_SAVE = {**BTN, 'backgroundColor': '#238636', 'borderColor': '#2ea043', 'color': '#fff'}

_default_skeleton = GLB_FILES[0]
_init_trg_elems, _init_layout = build_target_elements(_default_skeleton)

app = dash.Dash(__name__)
app.title = "Skeleton Annotator"

app.layout = html.Div(
    style={
        'backgroundColor': '#0d1117', 'height': '100vh', 'width': '100vw',
        'margin': 0, 'padding': 0, 'display': 'flex', 'flexDirection': 'column',
        'overflow': 'hidden', 'fontFamily': 'monospace',
    },
    children=[
        # ── header ────────────────────────────────────────────────────────────
        html.Div(style=HEADER_STYLE, children=[
            html.Span("Skeleton Annotator",
                      style={'color': '#58a6ff', 'fontSize': '13px', 'fontWeight': 'bold',
                             'flexShrink': 0}),
            dcc.Dropdown(
                id='skeleton-dropdown',
                options=[{'label': f, 'value': f} for f in GLB_FILES],
                value=_default_skeleton,
                clearable=False,
                style={'width': '220px', 'fontSize': '12px', 'flexShrink': 0},
            ),
            dcc.Textarea(
                id='description-area',
                placeholder='Skeleton description…',
                style={
                    'flex': 1, 'height': '32px', 'backgroundColor': '#21262d',
                    'color': '#c9d1d9', 'border': '1px solid #30363d',
                    'borderRadius': '6px', 'padding': '4px 8px',
                    'fontSize': '12px', 'fontFamily': 'monospace', 'resize': 'none',
                },
            ),
            html.Button("Save", id='save-btn', n_clicks=0, style=BTN_SAVE),
            html.Button("Clear", id='clear-btn', n_clicks=0, style=BTN),
            html.Span(id='status-text',
                      style={'color': '#8b949e', 'fontSize': '11px', 'flexShrink': 0}),
        ]),

        # ── main area ─────────────────────────────────────────────────────────
        html.Div(
            style={'display': 'flex', 'flex': 1, 'overflow': 'hidden'},
            children=[
                # Left: IK rig
                html.Div(style=PANEL_STYLE, children=[
                    html.Div(
                        "IK RIG  ·  H36M joints 0-16  ·  click to select source",
                        style=PANEL_LABEL_STYLE,
                    ),
                    cyto.Cytoscape(
                        id='ik-graph',
                        elements=build_ik_elements(),
                        layout={'name': 'preset', 'fit': True, 'padding': 40},
                        stylesheet=STYLESHEET,
                        style={'width': '100%', 'height': '100%',
                               'backgroundColor': '#0d1117'},
                    ),
                ]),
                html.Div(style=DIVIDER_STYLE),
                # Right: target skeleton
                html.Div(style=PANEL_STYLE, children=[
                    html.Div(id='target-label', style=PANEL_LABEL_STYLE,
                             children="TARGET  ·  click node to assign selected IK joint"),
                    cyto.Cytoscape(
                        id='target-graph',
                        elements=_init_trg_elems,
                        layout=_init_layout,
                        stylesheet=STYLESHEET,
                        style={'width': '100%', 'height': '100%',
                               'backgroundColor': '#0d1117'},
                    ),
                ]),
                html.Div(style=DIVIDER_STYLE),
                # Right sidebar: assignment table
                html.Div(
                    style={'width': '270px', 'display': 'flex', 'flexDirection': 'column',
                           'overflowY': 'auto', 'backgroundColor': '#0d1117', 'flexShrink': 0},
                    children=[
                        html.Div("ASSIGNMENTS  (IK# · name → target)",
                                 style=PANEL_LABEL_STYLE),
                        html.Div(
                            id='assignment-table',
                            children=make_assignment_table({}, _default_skeleton),
                            style={'overflowY': 'auto', 'flex': 1},
                        ),
                    ],
                ),
            ],
        ),

        # ── state stores ──────────────────────────────────────────────────────
        dcc.Store(id='selected-ik', data=None),
        dcc.Store(id='assignments', data={}),
        dcc.Store(id='annotations-store', data=INITIAL_ANNOTATIONS),
    ],
)


# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('selected-ik', 'data'),
    Input('ik-graph', 'tapNodeData'),
    prevent_initial_call=True,
)
def select_ik_joint(tap_data):
    if tap_data is None:
        return no_update
    return tap_data['id']   # str node id of the clicked IK joint


@app.callback(
    Output('assignments', 'data'),
    Output('description-area', 'value'),
    Input('skeleton-dropdown', 'value'),
    Input('target-graph', 'tapNodeData'),
    Input('clear-btn', 'n_clicks'),
    State('selected-ik', 'data'),
    State('assignments', 'data'),
    State('annotations-store', 'data'),
    prevent_initial_call=True,
)
def update_assignments(skeleton, tap_data, _clear, selected_ik, assignments, annotations):
    triggered = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered == 'skeleton-dropdown':
        if skeleton and annotations and skeleton in annotations:
            entry = annotations[skeleton]
            loaded = names_to_assignments(skeleton, entry.get('assignments', {}))
            return loaded, entry.get('description', '')
        return {}, ''

    if triggered == 'clear-btn':
        return {}, no_update

    if triggered == 'target-graph':
        if tap_data is None or selected_ik is None:
            return no_update, no_update
        target_id = tap_data['id']
        ik_node_id = int(selected_ik)
        if ik_node_id not in IK_IDX_OF:
            return no_update, no_update
        ik_idx_str = str(IK_IDX_OF[ik_node_id])
        new_assignments = {k: v for k, v in assignments.items() if v != target_id}
        new_assignments[ik_idx_str] = target_id
        return new_assignments, no_update

    return no_update, no_update


@app.callback(
    Output('ik-graph', 'elements'),
    Output('target-graph', 'elements'),
    Output('target-graph', 'layout'),
    Output('assignment-table', 'children'),
    Output('target-label', 'children'),
    Input('assignments', 'data'),
    Input('selected-ik', 'data'),
    State('skeleton-dropdown', 'value'),
)
def render_graphs(assignments, selected_ik, skeleton):
    if assignments is None:
        assignments = {}

    assigned_ik_node_ids = {
        str(IK_ORDERED[int(k)]) for k in assignments if 0 <= int(k) < N_IK
    }

    ik_elems = build_ik_elements(
        selected_ik_id=selected_ik,
        assigned_ik_node_ids=assigned_ik_node_ids,
    )
    trg_elems, layout = build_target_elements(skeleton, assignments) if skeleton else ([], {})
    table = make_assignment_table(assignments, skeleton)

    n_assigned = len(assignments)
    label = (
        f"TARGET  ·  {skeleton or '—'}  ·  {n_assigned}/17 assigned  ·  "
        "click node to assign selected IK joint"
    )
    return ik_elems, trg_elems, layout, table, label


@app.callback(
    Output('annotations-store', 'data'),
    Output('status-text', 'children'),
    Input('save-btn', 'n_clicks'),
    State('assignments', 'data'),
    State('description-area', 'value'),
    State('skeleton-dropdown', 'value'),
    State('annotations-store', 'data'),
    prevent_initial_call=True,
)
def save_annotation(_, assignments, description, skeleton, annotations):
    if not skeleton:
        return no_update, "No skeleton selected."
    name_assignments = assignments_to_names(skeleton, assignments or {})
    updated = dict(annotations or {})
    updated[skeleton] = {
        'description': description or '',
        'assignments': name_assignments,
    }
    try:
        save_annotations(updated)
        n = len(name_assignments)
        return updated, f"Saved {n}/17 for {skeleton}"
    except Exception as e:
        return no_update, f"Error saving: {e}"


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = 8052
    url = f'http://127.0.0.1:{port}'

    server_thread = threading.Thread(
        target=lambda: app.run(host='127.0.0.1', debug=False, use_reloader=False, port=port),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)
    webbrowser.open(url)

    print(f"\nAnnotator running at {url}")
    print("Press q + Enter to quit.\n")

    while True:
        try:
            if input().strip().lower() == 'q':
                break
        except (EOFError, KeyboardInterrupt):
            break

    os.kill(os.getpid(), signal.SIGINT)
