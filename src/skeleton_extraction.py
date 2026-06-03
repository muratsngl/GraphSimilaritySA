import pygltflib
import networkx as nx
import numpy as np

def build_qap_ready_graph(glb_path):
    print(f"\n--- Extracting Skeleton from: {glb_path} ---")
    
    gltf = pygltflib.GLTF2().load(glb_path)
    if not gltf.skins:
        raise ValueError(f"No skin/skeleton found in {glb_path}.")
        
    skin = gltf.skins[0]
    joints = set(skin.joints) 
    
    # CRITICAL FIX: Use a Directed Graph to preserve parent -> child relationships
    G = nx.DiGraph() 
    
    for joint_index in joints:
        node = gltf.nodes[joint_index]
        node_name = node.name if node.name else f"Joint_{joint_index}"
        G.add_node(joint_index, name=node_name)
        
        if node.children:
            for child_index in node.children:
                if child_index in joints:
                    child_node = gltf.nodes[child_index]
                    
                    if child_node.translation:
                        tx, ty, tz = child_node.translation
                        bone_length = np.sqrt(tx**2 + ty**2 + tz**2)
                    else:
                        bone_length = 0.0 
                        
                    G.add_edge(joint_index, child_index, weight=bone_length)

    print(f"Graph initialized: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G