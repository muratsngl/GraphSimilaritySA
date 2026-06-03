import networkx as nx

def prune_leaves_iteratively(G):
    """
    Leaf-Driven Collapse: Iterates exclusively over the graph's leaf nodes exactly once.
    Traces each leaf backward to its nearest junction and removes that specific path, 
    effectively collapsing the entire appendage into the junction.
    """
    print(f"\n--- Starting Leaf-Driven Appendage Collapse ---")
    print(f"Initial size: {G.number_of_nodes()} nodes.")
    
    pruned_G = G.copy()
    
    # 1. Identify all absolute end-points (leaves) and splits (junctions)
    leaves = [n for n, d in pruned_G.out_degree() if d == 0]
    junctions = set([n for n, d in pruned_G.out_degree() if d > 1])
    
    nodes_to_remove = set()
    
    # 2. Iterate through the leaves exactly ONCE
    for leaf in leaves:
        current_node = leaf
        path_to_collapse = []
        
        # 3. Trace backward from the leaf
        while True:
            parents = list(pruned_G.predecessors(current_node))
            
            # Safety break if we somehow hit the root without finding a junction
            if not parents:
                break 
                
            parent = parents[0]
            
            # If the parent is a junction (like a wrist or ankle), we stop tracing.
            if parent in junctions:
                path_to_collapse.append(current_node)
                nodes_to_remove.update(path_to_collapse)
                break
                
            # Otherwise, it's an intermediate bone (like a knuckle). Add it and keep tracing up.
            path_to_collapse.append(current_node)
            current_node = parent

    # 4. Execute the collapse in one single, clean sweep
    if nodes_to_remove:
        print(f"Collapsing {len(leaves)} leaf-paths ({len(nodes_to_remove)} total bones) into their junctions...")
        pruned_G.remove_nodes_from(nodes_to_remove)
        print("Leaf collapse complete.")
    else:
        print("No paths found to collapse.")
        
    print(f"Final pruned size: {pruned_G.number_of_nodes()} nodes.")
    
    # Recalculate sorted nodes to perfectly align with the future QAP matrix
    sorted_pruned_nodes = sorted(pruned_G.nodes())
    
    return pruned_G, sorted_pruned_nodes