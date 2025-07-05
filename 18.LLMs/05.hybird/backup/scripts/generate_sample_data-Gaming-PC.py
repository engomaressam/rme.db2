"""
Generate Sample Purchase Order Data for Testing
"""
import sys
import os
import json
import random
from datetime import datetime, timedelta
from loguru import logger

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.graphrag import GraphRAG

# Sample data for generating realistic purchase orders
PROJECTS = [
    {"name": "Al-ASEMA Bridge", "code": "ASB-2023"},
    {"name": "Riyadh Metro Line 3", "code": "RML3-2022"},
    {"name": "Jeddah Airport Expansion", "code": "JAE-2024"},
    {"name": "NEOM Smart City Phase 1", "code": "NSC1-2023"},
    {"name": "Red Sea Development", "code": "RSD-2022"}
]

SUPPLIERS = [
    {"name": "Modern Engineering", "id": "SUP-001", "category": "Construction"},
    {"name": "Rowad Technical Services", "id": "SUP-002", "category": "Technical"},
    {"name": "Al-Faisal Trading Co.", "id": "SUP-003", "category": "Trading"},
    {"name": "Saudi Steel Industries", "id": "SUP-004", "category": "Manufacturing"},
    {"name": "Global Concrete Solutions", "id": "SUP-005", "category": "Materials"}
]

ITEMS = [
    {"name": "Concrete (Grade 40)", "unit": "m³", "min_value": 500, "max_value": 1500},
    {"name": "Steel Reinforcement Bars", "unit": "ton", "min_value": 2000, "max_value": 5000},
    {"name": "Structural Steel Sections", "unit": "ton", "min_value": 3000, "max_value": 7000},
    {"name": "Excavation Works", "unit": "m³", "min_value": 200, "max_value": 800},
    {"name": "Backfilling Works", "unit": "m³", "min_value": 150, "max_value": 600},
    {"name": "Formwork", "unit": "m²", "min_value": 100, "max_value": 400},
    {"name": "Waterproofing Membrane", "unit": "m²", "min_value": 80, "max_value": 300},
    {"name": "Electrical Cables", "unit": "m", "min_value": 50, "max_value": 200},
    {"name": "HVAC Equipment", "unit": "unit", "min_value": 5000, "max_value": 20000},
    {"name": "Plumbing Fixtures", "unit": "set", "min_value": 1000, "max_value": 3000}
]

PAYMENT_TERMS = [
    "Net 30 days",
    "Net 60 days",
    "50% advance, 50% upon delivery",
    "30% advance, 70% upon completion",
    "As per contract terms"
]

DELIVERY_TERMS = [
    "Ex Works",
    "FOB Origin",
    "CIF Destination",
    "Delivered at Site",
    "As per project schedule"
]

def generate_po_number(project_code):
    """Generate a realistic purchase order number"""
    return f"{project_code}-PO-{random.randint(1000, 9999)}"

def generate_date(start_date=None, end_date=None):
    """Generate a random date within a range"""
    if not start_date:
        start_date = datetime.now() - timedelta(days=365)
    if not end_date:
        end_date = datetime.now()
    
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_number_of_days)
    
    return random_date.strftime("%Y-%m-%d")

def generate_items(min_items=2, max_items=6):
    """Generate a random list of items"""
    num_items = random.randint(min_items, max_items)
    selected_items = random.sample(ITEMS, num_items)
    
    items = []
    for item in selected_items:
        quantity = random.randint(1, 20)
        value = random.uniform(item["min_value"], item["max_value"])
        
        items.append({
            "name": item["name"],
            "quantity": quantity,
            "unit": item["unit"],
            "value": round(value, 2)
        })
    
    return items

def generate_purchase_order():
    """Generate a random purchase order"""
    project = random.choice(PROJECTS)
    supplier = random.choice(SUPPLIERS)
    items = generate_items()
    
    po_data = {
        "po_number": generate_po_number(project["code"]),
        "project": project,
        "date": generate_date(),
        "supplier": supplier,
        "items": items,
        "total_value": round(sum(item["value"] * item["quantity"] for item in items), 2),
        "payment_terms": random.choice(PAYMENT_TERMS),
        "delivery_terms": random.choice(DELIVERY_TERMS)
    }
    
    return po_data

def generate_sample_data(num_pos=20):
    """Generate sample purchase order data"""
    po_data_list = []
    
    for _ in range(num_pos):
        po_data = generate_purchase_order()
        po_data_list.append(po_data)
    
    return po_data_list

def save_sample_data(po_data_list, output_file):
    """Save sample data to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(po_data_list, f, indent=2)
        
        logger.info(f"Saved {len(po_data_list)} purchase orders to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving sample data: {str(e)}")
        return False

def import_to_graphrag(po_data_list):
    """
    Import purchase order data to GraphRAG system
    
    Args:
        po_data_list (list): List of purchase order data
        
    Returns:
        bool: Success or failure
    """
    try:
        # Initialize GraphRAG
        graphrag = GraphRAG(skip_llm=True)
        
        # Check if databases are connected
        if not hasattr(graphrag, 'neo4j') or not graphrag.neo4j.is_connected():
            logger.error("Neo4j not connected")
            return False
        
        if not hasattr(graphrag, 'qdrant') or not graphrag.qdrant.is_connected():
            logger.error("Qdrant not connected")
            return False
        
        # Import each PO
        for po_data in po_data_list:
            # Generate embedding
            embedding = graphrag.embedding_processor.generate_po_embedding(po_data)
            if not embedding:
                logger.error(f"Failed to generate embedding for PO: {po_data['po_number']}")
                continue
            
            # Add to Neo4j
            success = graphrag.neo4j.create_purchase_order(po_data)
            if success:
                logger.info(f"Added PO {po_data['po_number']} to Neo4j")
            else:
                logger.error(f"Failed to add PO {po_data['po_number']} to Neo4j")
            
            # Add to Qdrant
            success = graphrag.qdrant.add_purchase_order(po_data['po_number'], embedding, po_data)
            if success:
                logger.info(f"Added PO {po_data['po_number']} to Qdrant")
            else:
                logger.error(f"Failed to add PO {po_data['po_number']} to Qdrant")
        
        # Build graph from data
        graphrag.build_graph_from_data(po_data_list)
        
        # Save graph to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_file = os.path.join(config.OUTPUT_DIRECTORY, f"graph_{timestamp}.json")
        graphrag.save_graph_to_file(graph_file)
        
        # Visualize graph
        output_file = os.path.join(config.OUTPUT_DIRECTORY, f"graph_visualization_{timestamp}.png")
        graphrag.visualize_graph(output_file, show=False)
        
        # Close connections
        graphrag.close()
        
        logger.info(f"Successfully imported {len(po_data_list)} purchase orders to GraphRAG system")
        logger.info(f"Graph saved to {graph_file}")
        logger.info(f"Graph visualization saved to {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error importing to GraphRAG: {str(e)}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Sample Purchase Order Data for Testing")
    parser.add_argument("--num-pos", type=int, default=20,
                        help="Number of purchase orders to generate")
    parser.add_argument("--output-file", type=str, 
                        default=os.path.join(config.OUTPUT_DIRECTORY, "sample_pos.json"),
                        help="Output file for sample data")
    parser.add_argument("--import", action="store_true", default=False,
                        help="Import sample data to GraphRAG system")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Generate sample data
    po_data_list = generate_sample_data(args.num_pos)
    
    # Save sample data
    success = save_sample_data(po_data_list, args.output_file)
    if not success:
        return
    
    # Import to GraphRAG if requested
    if getattr(args, 'import'):
        success = import_to_graphrag(po_data_list)
        
        if success:
            print(f"Successfully imported {len(po_data_list)} purchase orders to GraphRAG system")
        else:
            print("Failed to import purchase orders to GraphRAG system")
    else:
        print(f"Generated {len(po_data_list)} sample purchase orders")
        print(f"Saved to {args.output_file}")
        print("To import to GraphRAG system, run with --import flag")

if __name__ == "__main__":
    main()
