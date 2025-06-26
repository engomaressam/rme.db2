"""
Search Purchase Orders using GraphRAG Hybrid System
"""
import sys
import os
import argparse
from loguru import logger

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.graphrag import GraphRAG
from src.processors.embedding_processor import EmbeddingProcessor

def search_purchase_orders(query, limit=5, project_name=None, supplier_name=None):
    """
    Search for purchase orders using GraphRAG hybrid system
    
    Args:
        query (str): Search query
        limit (int): Maximum number of results
        project_name (str): Filter by project name
        supplier_name (str): Filter by supplier name
        
    Returns:
        list: Search results
    """
    try:
        # Initialize GraphRAG
        graphrag = GraphRAG(skip_llm=True)
        
        # Check if databases are connected
        if not hasattr(graphrag, 'neo4j') or not graphrag.neo4j.is_connected():
            logger.error("Neo4j not connected")
            return None
        
        if not hasattr(graphrag, 'qdrant') or not graphrag.qdrant.is_connected():
            logger.error("Qdrant not connected")
            return None
        
        # Search for similar purchase orders
        results = graphrag.search_similar_pos(query, limit, project_name, supplier_name)
        
        # Close connections
        graphrag.close()
        
        return results
    except Exception as e:
        logger.error(f"Error searching purchase orders: {str(e)}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Search Purchase Orders using GraphRAG Hybrid System")
    parser.add_argument("--query", "-q", type=str, required=True,
                        help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=5,
                        help="Maximum number of results")
    parser.add_argument("--project", "-p", type=str, default=None,
                        help="Filter by project name")
    parser.add_argument("--supplier", "-s", type=str, default=None,
                        help="Filter by supplier name")
    
    args = parser.parse_args()
    
    # Search purchase orders
    results = search_purchase_orders(args.query, args.limit, args.project, args.supplier)
    
    if results:
        print(f"Found {len(results)} purchase orders:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. PO: {result['po_number']}")
            print(f"   Project: {result['project_name']}")
            print(f"   Supplier: {result['supplier_name']}")
            print(f"   Date: {result['date']}")
            print(f"   Total Value: {result['total_value']}")
            print(f"   Similarity Score: {result['score']:.4f}")
    else:
        print("No purchase orders found or database connection failed")

if __name__ == "__main__":
    main()
