"""
Visualize Purchase Order Graph
"""
import sys
import os
import argparse
from loguru import logger
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.graphrag import GraphRAG

def visualize_graph(graph_file, output_file=None, show=True):
    """
    Visualize a graph from a JSON file
    
    Args:
        graph_file (str): Path to graph JSON file
        output_file (str): Path to output file
        show (bool): Show the graph
        
    Returns:
        str: Path to output file
    """
    try:
        # Initialize GraphRAG in visualization-only mode
        graphrag = GraphRAG(visualization_only=True)
        
        # Load graph from file
        if not graphrag.load_graph_from_file(graph_file):
            logger.error(f"Failed to load graph from {graph_file}")
            return None
        
        # Create output file if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(config.OUTPUT_DIRECTORY, f"graph_visualization_{timestamp}.png")
        
        # Visualize graph
        output_path = graphrag.visualize_graph(output_file, show)
        
        return output_path
    except Exception as e:
        logger.error(f"Error visualizing graph: {str(e)}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Visualize Purchase Order Graph")
    parser.add_argument("--graph-file", "-g", type=str, required=True,
                        help="Path to graph JSON file")
    parser.add_argument("--output-file", "-o", type=str, default=None,
                        help="Path to output file")
    parser.add_argument("--no-show", "-n", action="store_true", default=False,
                        help="Don't show the graph")
    
    args = parser.parse_args()
    
    # Visualize graph
    output_file = visualize_graph(args.graph_file, args.output_file, not args.no_show)
    
    if output_file:
        print(f"Successfully visualized graph. Saved to {output_file}")
    else:
        print("Failed to visualize graph")

if __name__ == "__main__":
    main()
