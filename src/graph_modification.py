import networkx as nx


def _tag_c2_descendants(G):
    """
    Pre-pass: returns the set of nodes that have at least one c>2 (out_degree > 2)
    node anywhere in their subtree. Processed bottom-up via reverse topological order
    so each node's children are already resolved before the node itself is evaluated.
    """
    has_c2_descendant = set()

    for node in reversed(list(nx.topological_sort(G))):
        for child in G.successors(node):
            if G.out_degree(child) > 2 or child in has_c2_descendant:
                has_c2_descendant.add(node)
                break

    return has_c2_descendant


def prune_leaves_iteratively(G):
    """
    DFS-based appendage pruning with terminal-child majority gate.

    Pre-pass: tags every node as having a c>2 descendant or not.

    Pruning DFS: a c>2 node is only registered as an anchor if it has >= 2 children
    whose subtrees are purely terminal (no c>2 node anywhere inside them).
    This distinguishes true fan-out endpoints (wrists with 5 finger chains) from
    structural branch points (upper torso, pelvis) where most children lead into
    complex sub-skeletons. Roots are never registered as anchors regardless.

    - Anchor node       →  becomes last_c2 for all its children.
    - Leaf + anchor     →  backwards-walk from leaf to anchor, every node in between
                           is marked for removal (anchor itself is kept).
    - Leaf + no anchor  →  left untouched (e.g. head at end of neck chain).
    - Intermediate node →  propagates the current anchor unchanged.

    All removals execute in one sweep after the DFS completes.
    """
    print(f"\n--- Starting DFS-Based Appendage Pruning ---")
    print(f"Initial size: {G.number_of_nodes()} nodes.")

    pruned_G = G.copy()

    roots = [n for n, d in pruned_G.in_degree() if d == 0]
    if not roots:
        print("No root found. Returning graph unchanged.")
        return pruned_G, sorted(pruned_G.nodes())

    has_c2_descendant = _tag_c2_descendants(pruned_G)

    nodes_to_remove = set()

    # Stack holds (node, last_c2_ancestor)
    stack = [(root, None) for root in roots]
    visited = set()

    while stack:
        node, last_c2 = stack.pop()

        if node in visited:
            continue
        visited.add(node)

        children = list(pruned_G.successors(node))
        out_deg = len(children)

        if out_deg > 2 and node not in roots:
            # Count children whose subtrees contain no further c>2 node
            terminal_child_count = sum(
                1 for c in children
                if pruned_G.out_degree(c) <= 2 and c not in has_c2_descendant
            )

            if terminal_child_count >= 2:
                # True fan-out endpoint (e.g. wrist) — register as anchor
                for child in children:
                    stack.append((child, node))
            else:
                # Structural branch point (e.g. upper torso) — pass anchor through
                for child in children:
                    stack.append((child, last_c2))

        elif out_deg == 0:
            # Leaf — prune back to nearest anchor if one exists
            if last_c2 is not None:
                curr = node
                while curr != last_c2:
                    nodes_to_remove.add(curr)
                    parents = list(pruned_G.predecessors(curr))
                    if not parents:
                        break
                    curr = parents[0]
            # else: no anchor above this leaf — leave it alone

        else:
            # Intermediate node (1 or 2 children) — propagate anchor unchanged
            for child in children:
                stack.append((child, last_c2))

    if nodes_to_remove:
        print(f"Removing {len(nodes_to_remove)} nodes from appendage chains...")
        pruned_G.remove_nodes_from(nodes_to_remove)
        print("Pruning complete.")
    else:
        print("No nodes to prune.")

    print(f"Final pruned size: {pruned_G.number_of_nodes()} nodes.")

    return pruned_G, sorted(pruned_G.nodes())
