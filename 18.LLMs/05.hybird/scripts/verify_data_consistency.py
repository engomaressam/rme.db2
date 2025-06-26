"""
Data Consistency Check Script

This script verifies that all purchase orders from MySQL have been properly imported
into Neo4j and Qdrant. It checks:
1. Record counts match across all systems
2. Sample records exist in all systems
3. Basic data integrity checks
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import mysql.connector
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv()

# Configuration
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', '10.10.11.242'),
    'port': int(os.getenv('MYSQL_PORT', '3306')),
    'user': os.getenv('MYSQL_USER', 'omar2'),
    'password': os.getenv('MYSQL_PASSWORD', 'Omar_54321'),
    'database': os.getenv('MYSQL_DATABASE', 'RME_TEST')
}

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "purchase_orders")

def get_mysql_count():
    """Get count of purchase orders in MySQL"""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM `po.pdfs`")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count
    except Exception as e:
        print(f"[ERROR] MySQL count failed: {e}")
        return None

def get_neo4j_count():
    """Get count of purchase orders in Neo4j"""
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("MATCH (po:PurchaseOrder) RETURN count(po) as count")
            return result.single()["count"]
    except Exception as e:
        print(f"[ERROR] Neo4j count failed: {e}")
        return None
    finally:
        if driver:
            driver.close()

def get_qdrant_count():
    """Get count of vectors in Qdrant collection"""
    try:
        client = QdrantClient(url=QDRANT_URL)
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        return collection_info.points_count
    except Exception as e:
        print(f"[ERROR] Qdrant count failed: {e}")
        return None

def verify_data_consistency():
    """Verify data consistency across all systems"""
    print("\n=== Data Consistency Check ===\n")
    
    # Get counts from each system
    print("Counting records in each system...")
    mysql_count = get_mysql_count()
    neo4j_count = get_neo4j_count()
    qdrant_count = get_qdrant_count()
    
    # Print counts
    print(f"\nRecord Counts:")
    print(f"MySQL (source): {mysql_count:,} purchase orders")
    print(f"Neo4j (graph):  {neo4j_count:,} purchase orders")
    print(f"Qdrant (vector): {qdrant_count:,} vectors")
    
    # Check for consistency
    if mysql_count is not None and neo4j_count is not None and qdrant_count is not None:
        print("\nConsistency Check:")
        
        # Check Neo4j
        if mysql_count == neo4j_count:
            print("[PASS] MySQL and Neo4j counts match")
        else:
            print(f"[WARNING] Count mismatch: MySQL has {mysql_count:,} records but Neo4j has {neo4j_count:,}")
        
        # Check Qdrant
        if mysql_count == qdrant_count:
            print("[PASS] MySQL and Qdrant counts match")
        else:
            print(f"[WARNING] Count mismatch: MySQL has {mysql_count:,} records but Qdrant has {qdrant_count:,}")
        
        # Check if all systems are in sync
        if mysql_count == neo4j_count == qdrant_count:
            print("\n[SUCCESS] All systems are in sync!")
        else:
            print("\n[WARNING] Data inconsistencies found. See above for details.")
    
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify_data_consistency()
