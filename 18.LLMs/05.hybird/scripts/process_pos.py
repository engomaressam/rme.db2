"""
Process Purchase Orders from Subfolders
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

def process_directory(directory, recursive=True, visualization_only=False, skip_llm=True):
    """
    Process all PDF files in a directory and its subdirectories
    
    Args:
        directory (str): Directory to process
        recursive (bool): Process subdirectories
        visualization_only (bool): Run in visualization-only mode
        skip_llm (bool): Skip LLM initialization
        
    Returns:
        str: Path to output graph file
    """
    try:
        # Initialize GraphRAG
        graphrag = GraphRAG(skip_llm=skip_llm, visualization_only=visualization_only)
        logger.info(f"Processing directory: {directory}")
        
        if visualization_only:
            # Load existing graph if available
            graph_file = os.path.join(config.OUTPUT_DIRECTORY, "graph.json")
            if os.path.isfile(graph_file):
                graphrag.load_graph_from_file(graph_file)
                logger.info(f"Loaded existing graph from {graph_file}")
            else:
                logger.warning(f"No existing graph found at {graph_file}")
                return None
        else:
            # Process PDFs and build graph
            po_data_list = graphrag.process_directory(directory, recursive)
            
            if not po_data_list:
                logger.warning("No purchase orders found")
                return None
            
            # Build graph from data
            graphrag.build_graph_from_data(po_data_list)
            
            # Save graph to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_file = os.path.join(config.OUTPUT_DIRECTORY, f"graph_{timestamp}.json")
            graphrag.save_graph_to_file(graph_file)
        
        # Visualize graph
        output_file = os.path.join(config.OUTPUT_DIRECTORY, f"graph_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        graphrag.visualize_graph(output_file, show=False)
        
        # Close connections
        graphrag.close()
        
        return output_file
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Process Purchase Orders from Subfolders")
    parser.add_argument("--directory", "-d", type=str, default=config.PDF_DIRECTORY,
                        help="Directory containing PDF files")
    parser.add_argument("--recursive", "-r", action="store_true", default=True,
                        help="Process subdirectories")
    parser.add_argument("--visualization-only", "-v", action="store_true", default=False,
                        help="Run in visualization-only mode")
    parser.add_argument("--with-llm", "-l", action="store_true", default=False,
                        help="Initialize LLM (default: skip)")
    
    args = parser.parse_args()
    
    # Process directory
    output_file = process_directory(
        args.directory,
        args.recursive,
        args.visualization_only,
        not args.with_llm
    )
    
    if output_file:
        print(f"Successfully processed purchase orders. Graph visualization saved to {output_file}")
    else:
        print("Failed to process purchase orders")

if __name__ == "__main__":
    main()
