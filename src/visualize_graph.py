import sys
import os
import signal
import socket
import threading
import webbrowser
import time
sys.path.insert(0, os.path.dirname(__file__))

import dash
from dash import html
import dash_cytoscape as cyto

from main import run_pipeline

GLB_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'horse_riggedgame_ready.glb')

print("Running pipeline...")
try:
    raw_G, pruned_G, _, _ = run_pipeline(GLB_PATH)
    print(f"Pipeline done — raw: {raw_G.number_of_nodes()} nodes, pruned: {pruned_G.number_of_nodes()} nodes")
except Exception as e:
    print(f"Pipeline failed: {e}")
    sys.exit(1)


def graph_to_cytoscape_elements(G):
    elements = []
    for node, data in G.nodes(data=True):
        elements.append({
            'data': {
                'id': str(node),
                'label': data.get('name', str(node))
            }
        })
    for source, target, data in G.edges(data=True):
        elements.append({
            'data': {
                'source': str(source),
                'target': str(target),
                'weight': round(data.get('weight', 0.0), 4)
            }
        })
    return elements


def get_roots(G):
    return [str(n) for n, d in G.in_degree() if d == 0]


STYLESHEET = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'color': '#c9d1d9',
            'background-color': '#1f6feb',
            'border-color': '#58a6ff',
            'border-width': 2,
            'font-size': '11px',
            'text-valign': 'center',
            'text-halign': 'right',
            'text-margin-x': 8,
            'text-background-color': '#0d1117',
            'text-background-opacity': 0.7,
            'text-background-padding': '2px',
            'width': 28,
            'height': 28,
        }
    },
    {
        'selector': 'edge',
        'style': {
            'line-color': '#3d444d',
            'target-arrow-color': '#3d444d',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 0.8,
            'curve-style': 'bezier',
            'width': 2,
        }
    },
    {
        'selector': ':selected',
        'style': {
            'background-color': '#f78166',
            'border-color': '#ff7b72',
            'border-width': 3,
            'line-color': '#f78166',
            'target-arrow-color': '#f78166',
        }
    }
]

PANEL_LABEL_STYLE = {
    'color': '#8b949e',
    'fontSize': '12px',
    'fontFamily': 'monospace',
    'padding': '8px 14px',
    'borderBottom': '1px solid #21262d',
    'backgroundColor': '#161b22',
    'flexShrink': 0,
}

DIVIDER_STYLE = {
    'width': '1px',
    'backgroundColor': '#21262d',
    'flexShrink': 0,
}

PANEL_STYLE = {
    'flex': 1,
    'display': 'flex',
    'flexDirection': 'column',
    'overflow': 'hidden',
}

def make_panel(graph_id, G, label):
    roots = get_roots(G)
    return html.Div(
        style=PANEL_STYLE,
        children=[
            html.Div(label, style=PANEL_LABEL_STYLE),
            cyto.Cytoscape(
                id=graph_id,
                elements=graph_to_cytoscape_elements(G),
                layout={
                    'name': 'breadthfirst',
                    'directed': True,
                    'roots': roots,
                    'spacingFactor': 1.75,
                    'animate': False,
                    'padding': 40,
                },
                stylesheet=STYLESHEET,
                style={
                    'width': '100%',
                    'height': '100%',
                    'backgroundColor': '#0d1117',
                }
            ),
        ]
    )


app = dash.Dash(__name__)
app.title = "Skeleton Graph Viewer"

app.layout = html.Div(
    style={
        'backgroundColor': '#0d1117',
        'height': '100vh',
        'width': '100vw',
        'margin': 0,
        'padding': 0,
        'display': 'flex',
        'flexDirection': 'column',
        'overflow': 'hidden',
    },
    children=[
        # Top header bar
        html.Div(
            style={
                'padding': '8px 20px',
                'borderBottom': '1px solid #21262d',
                'backgroundColor': '#161b22',
                'flexShrink': 0,
            },
            children=html.Span(
                "Skeleton Graph Viewer",
                style={'color': '#58a6ff', 'fontSize': '13px', 'fontFamily': 'monospace', 'fontWeight': 'bold'}
            )
        ),

        # Side-by-side panels
        html.Div(
            style={'display': 'flex', 'flex': 1, 'overflow': 'hidden'},
            children=[
                make_panel(
                    'raw-graph',
                    raw_G,
                    f"RAW  ·  {raw_G.number_of_nodes()} nodes  ·  {raw_G.number_of_edges()} edges"
                ),
                html.Div(style=DIVIDER_STYLE),
                make_panel(
                    # --- SA real-time hook: update 'pruned-graph' elements via
                    # a dcc.Store + dcc.Interval callback to stream SA iterations.
                    'pruned-graph',
                    pruned_G,
                    f"PRUNED  ·  {pruned_G.number_of_nodes()} nodes  ·  {pruned_G.number_of_edges()} edges"
                ),
            ]
        )
    ]
)


if __name__ == '__main__':
    with socket.socket() as _s:
        _s.bind(('', 0))
        port = _s.getsockname()[1]
    url = f'http://127.0.0.1:{port}'

    server_thread = threading.Thread(
        target=lambda: app.run(host='127.0.0.1', debug=False, use_reloader=False, port=port),
        daemon=True
    )
    server_thread.start()

    # Wait for the server to be ready before opening the browser
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
