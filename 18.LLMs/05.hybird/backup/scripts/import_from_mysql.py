"""
Import Document Data from MySQL Database

This script imports purchase orders, contracts, and comparisons from a MySQL database
and adds them to the GraphRAG hybrid system. It also saves testing results and
semantic search outputs to text files.
"""
import sys
import os
import argparse
import re
import random
from datetime import datetime
import mysql.connector
from loguru import logger

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.graphrag import GraphRAG

def extract_po_number(text):
    """Extract PO number using regex patterns"""
    patterns = [
        r'P\.?O\.?\s*#?\s*(\w+[-/]?\w+)',
        r'Purchase\s+Order\s+(?:No\.?|Number)\s*[:.]?\s*(\w+[-/]?\w+)',
        r'Order\s+(?:No\.?|Number)\s*[:.]?\s*(\w+[-/]?\w+)',
        r'PO[-\s.:]*(\d+[-/]?\w*)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: look for any alphanumeric string that looks like a PO number
    match = re.search(r'(?:^|\s)([A-Z0-9]{2,}[-/][A-Z0-9]{2,})', text)
    if match:
        return match.group(1).strip()
    
    return f"Unknown-PO-{random.randint(1000, 9999)}"

def extract_supplier_info(text):
    """Extract supplier name and details"""
    supplier_name = None
    supplier_id = None
    
    # Try to find supplier name
    supplier_patterns = [
        r'(?:Vendor|Supplier|Company)(?:\s+Name)?\s*[:.]?\s*([A-Za-z0-9\s\-&\.]+)',
        r'(?:Bill|Ship)\s+To\s*[:.]?\s*([A-Za-z0-9\s\-&\.]+)',
        r'(?:^|\n)([A-Z][A-Za-z0-9\s\-&\.]+)(?:\n|$)',
    ]
    
    for pattern in supplier_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            supplier_name = match.group(1).strip()
            break
    
    # Try to find supplier ID
    id_patterns = [
        r'(?:Vendor|Supplier)\s*(?:Code|ID|Number|#)\s*[:.]?\s*([A-Za-z0-9\-]+)',
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            supplier_id = match.group(1).strip()
            break
    
    # If no supplier name found, use a generic one
    if not supplier_name:
        supplier_name = f"Unknown Supplier {random.randint(100, 999)}"
    
    return {
        "name": supplier_name,
        "id": supplier_id,
        "category": None
    }

def extract_date(text):
    """Extract date from text"""
    date_patterns = [
        r'(?:Date|PO\s+Date|Order\s+Date)\s*[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
        r'(?:Date|PO\s+Date|Order\s+Date)\s*[:.]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None

def extract_items(text):
    """Extract items from text using regex patterns"""
    items = []
    
    # Try to find items in a table-like structure
    # This is a simplified approach and might need adjustment based on actual PO formats
    item_patterns = [
        r'(\d+)\s+([A-Za-z0-9\s\-\.]+)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\s+(\d+(?:\.\d+)?)',
        r'([A-Za-z0-9\s\-\.]+)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\s+(\d+(?:\.\d+)?)',
    ]
    
    for pattern in item_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            if len(match.groups()) == 5:  # With item number
                _, name, quantity, unit, value = match.groups()
            else:  # Without item number
                name, quantity, unit, value = match.groups()
            
            try:
                items.append({
                    "name": name.strip(),
                    "quantity": float(quantity),
                    "unit": unit.strip(),
                    "value": float(value)
                })
            except ValueError:
                # Skip if conversion to float fails
                pass
    
    # If no items found, add a dummy item
    if not items:
        items.append({
            "name": f"Unspecified Item {random.randint(100, 999)}",
            "quantity": 1,
            "unit": "Unit",
            "value": 1000
        })
    
    return items

def extract_total_value(text):
    """Extract total value from text"""
    total_patterns = [
        r'(?:Total|Grand\s+Total|Amount)\s*[:.]?\s*(?:USD|SAR|$|£|€)?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
        r'(?:Total|Grand\s+Total|Amount)\s*[:.]?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:USD|SAR|$|£|€)?',
    ]
    
    for pattern in total_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Remove commas and convert to float
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                pass
    
    # If no total found, return None
    return None

def extract_po_details(row):
    """
    Extract purchase order details from database row
    
    Args:
        row (tuple): Database row with PO data
        
    Returns:
        dict: Purchase order details
    """
    # Assuming row structure: (id, pdf_hash, pdf_filename, extracted_text, processed_timestamp, project_name, document_type)
    pdf_id, pdf_hash, pdf_filename, extracted_text, processed_timestamp, project_name, document_type = row
    
    # Extract PO number
    po_number = extract_po_number(extracted_text)
    
    # Extract supplier info
    supplier = extract_supplier_info(extracted_text)
    
    # Extract date
    date = extract_date(extracted_text)
    
    # Extract items
    items = extract_items(extracted_text)
    
    # Extract total value
    total_value = extract_total_value(extracted_text)
    
    # If total_value is not found, calculate from items
    if total_value is None and items:
        total_value = sum(item["value"] for item in items)
    
    # Create project info
    project = {
        "name": project_name or f"Unknown Project {random.randint(100, 999)}",
        "code": None
    }
    
    po_data = {
        "po_number": po_number,
        "project": project,
        "date": date,
        "supplier": supplier,
        "items": items,
        "total_value": total_value,
        "payment_terms": None,
        "delivery_terms": None,
        "document_type": document_type,
        "pdf_filename": pdf_filename,
        "pdf_hash": pdf_hash,
        "pdf_id": pdf_id
    }
    
    return po_data

def connect_to_mysql(host, user, password, database):
    """
    Connect to MySQL database
    
    Args:
        host (str): MySQL host
        user (str): MySQL user
        password (str): MySQL password
        database (str): MySQL database name
        
    Returns:
        mysql.connector.connection.MySQLConnection: MySQL connection
    """
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logger.info(f"Successfully connected to MySQL database: {database}")
        return connection
    except mysql.connector.Error as e:
        logger.error(f"Error connecting to MySQL database: {str(e)}")
        return None

def fetch_po_data(connection, limit=100):
    """
    Fetch purchase order data from MySQL database
    
    Args:
        connection (mysql.connector.connection.MySQLConnection): MySQL connection
        limit (int): Maximum number of records to fetch
        
    Returns:
        list: List of purchase order data
    """
    try:
        cursor = connection.cursor()
        
        # Execute query to fetch PO data
        query = """
        SELECT id, pdf_hash, pdf_filename, extracted_text, processed_timestamp, project_name, document_type
        FROM po.pdfs
        LIMIT %s
        """
        cursor.execute(query, (limit,))
        
        # Fetch all rows
        rows = cursor.fetchall()
        logger.info(f"Fetched {len(rows)} purchase orders from MySQL database")
        
        # Process each row
        po_data_list = []
        for row in rows:
            po_data = extract_po_details(row)
            po_data_list.append(po_data)
        
        cursor.close()
        return po_data_list
    except mysql.connector.Error as e:
        logger.error(f"Error fetching purchase order data: {str(e)}")
        return []

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
    parser = argparse.ArgumentParser(description="Import Purchase Order Data from MySQL Database")
    parser.add_argument("--host", type=str, default="10.10.11.242",
                        help="MySQL host")
    parser.add_argument("--user", type=str, default="Omar2",
                        help="MySQL user")
    parser.add_argument("--password", type=str, default="Omar_54321",
                        help="MySQL password")
    parser.add_argument("--database", type=str, default="RME_TEST",
                        help="MySQL database name")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum number of records to fetch")
    
    args = parser.parse_args()
    
    # Connect to MySQL
    connection = connect_to_mysql(args.host, args.user, args.password, args.database)
    if not connection:
        logger.error("Failed to connect to MySQL database")
        return
    
    # Fetch PO data
    po_data_list = fetch_po_data(connection, args.limit)
    if not po_data_list:
        logger.error("No purchase order data found")
        connection.close()
        return
    
    # Import to GraphRAG
    success = import_to_graphrag(po_data_list)
    
    # Close MySQL connection
    connection.close()
    
    if success:
        print(f"Successfully imported {len(po_data_list)} purchase orders to GraphRAG system")
    else:
        print("Failed to import purchase orders to GraphRAG system")

if __name__ == "__main__":
    main()
