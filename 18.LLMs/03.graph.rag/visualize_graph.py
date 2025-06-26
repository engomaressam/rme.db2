import json
import networkx as nx
import matplotlib.pyplot as plt
import os

def load_graph(graph_dir):
    """Load the knowledge graph from disk"""
    json_path = os.path.join(graph_dir, 'knowledge_graph.json')
    
    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # Create a new directed graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_nodes_from(graph_data['nodes'])
    
    # Add edges with attributes
    for u, v, d in graph_data['edges']:
        G.add_edge(u, v, **d)
    
    return G

def visualize_graph(G, output_dir):
    """Create a visualization of the graph using matplotlib"""
    plt.figure(figsize=(20, 20))
    
    # Use spring layout for node positioning
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
    
    # Draw edges with labels
    edge_labels = {(u, v): d.get('predicate', '') for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_edges(G, pos, arrows=True)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title("Knowledge Graph Visualization")
    plt.axis('off')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'graph_visualization.png'), format='PNG', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    graph_dir = r"C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\05.llm\graph.rag"
    
    # Load the graph
    print("Loading graph...")
    G = load_graph(graph_dir)
    
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create visualization
    print("Creating visualization...")
    visualize_graph(G, graph_dir)
    print(f"Visualization saved to {os.path.join(graph_dir, 'graph_visualization.png')}")
    
    # Print some graph statistics
    print("\nGraph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("\nMost connected nodes:")
    for node, degree in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {degree} connections")

if __name__ == "__main__":
    main()
