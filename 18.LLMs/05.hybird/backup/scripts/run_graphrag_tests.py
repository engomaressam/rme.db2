"""
Run GraphRAG System Tests and Save Results to Text File

This script generates sample purchase orders, imports them into the GraphRAG system,
and runs various semantic searches to test the system's functionality.
All test results are saved to a text file for review.
"""
import sys
import os
import json
import time
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.graphrag import GraphRAG
from scripts.generate_sample_data import generate_sample_data, import_to_graphrag

def run_semantic_search(graphrag, query, output_file):
    """Run a semantic search and save results to file"""
    print(f"\nRunning semantic search: '{query}'")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"\n\n=== Semantic Search: '{query}' ===\n")
        
        results = graphrag.search_similar_pos(query, limit=5)
        
        if not results:
            message = "No results found."
            print(message)
            f.write(f"{message}\n")
            return
        
        print(f"Found {len(results)} purchase orders:")
        f.write(f"Found {len(results)} purchase orders:\n")
        
        for i, result in enumerate(results, 1):
            po_info = (
                f"{i}. PO: {result.get('po_number', 'Unknown')}\n"
                f"   Project: {result.get('project_name', 'Unknown')}\n"
                f"   Supplier: {result.get('supplier_name', 'Unknown')}\n"
                f"   Date: {result.get('date', 'Unknown')}\n"
                f"   Total Value: {result.get('total_value', 'Unknown')}\n"
                f"   Similarity Score: {result.get('score', 0):.4f}\n"
            )
            print(po_info)
            f.write(po_info)

def main():
    """Main function"""
    # Create output directory if it doesn't exist
    os.makedirs(config.OUTPUT_DIRECTORY, exist_ok=True)
    
    # Create output file for test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(config.OUTPUT_DIRECTORY, f"graphrag_test_results_{timestamp}.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"GraphRAG System Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
    
    print(f"Generating 40 sample purchase orders...")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("Generating 40 sample purchase orders...\n")
    
    # Generate sample data
    po_data_list = generate_sample_data(num_pos=40)
    
    # Save sample data
    sample_file = os.path.join(config.OUTPUT_DIRECTORY, f"sample_pos_{timestamp}.json")
    with open(sample_file, "w", encoding="utf-8") as f_sample:
        json.dump(po_data_list, f_sample, indent=2)
    
    print(f"Generated 40 sample purchase orders and saved to {sample_file}")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"Generated 40 sample purchase orders and saved to {sample_file}\n")
    
    # Import to GraphRAG
    print(f"Importing purchase orders to GraphRAG system...")
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\nImporting purchase orders to GraphRAG system...\n")
    
    success = import_to_graphrag(po_data_list)
    
    if success:
        message = f"Successfully imported {len(po_data_list)} purchase orders to GraphRAG system"
        print(message)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
    else:
        message = "Failed to import purchase orders to GraphRAG system"
        print(message)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
        return
    
    # Initialize GraphRAG for searching
    graphrag = GraphRAG(skip_llm=True)
    
    # Run various semantic searches
    search_queries = [
        # Project and material related queries
        "steel reinforcement for bridge project",
        "concrete materials for NEOM project",
        "electrical equipment for airport expansion",
        "plumbing fixtures for metro line",
        
        # Terms and conditions related queries
        "payment terms with 50% advance payment",
        "delivery terms for on-site delivery",
        "net 30 days payment condition",
        "contracts with advance payment requirements",
        "purchase orders with specific delivery terms",
        "payment upon completion terms",
        
        # Supplier related queries
        "orders from Saudi Steel Industries",
        "high value orders from technical services suppliers",
        
        # Value related queries
        "purchase orders with value over 100000",
        "low cost material orders under 20000"
    ]
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n\n=== Running Semantic Searches ===\n")
    
    for query in search_queries:
        run_semantic_search(graphrag, query, output_file)
        time.sleep(1)  # Small delay between searches
    
    # Close connections
    graphrag.close()
    
    print(f"\nAll test results saved to {output_file}")

if __name__ == "__main__":
    main()
