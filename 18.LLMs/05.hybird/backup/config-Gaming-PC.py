"""
Configuration for GraphRAG Hybrid system
"""
import os
from dotenv import load_dotenv

# Load environment variables if .env file exists
if os.path.exists('.env'):
    load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag")

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # Only needed for cloud deployment

# Embedding Model Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Collection Configuration
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "purchase_orders")

# PDF Directory Configuration
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "D:\\OEssam\\01.pdfs")

# Output Directory Configuration
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "output")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Function to check if Neo4j is available
def check_neo4j_connection():
    """Check if Neo4j is available"""
    from neo4j import GraphDatabase
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            return result.single()["test"] == 1
    except Exception as e:
        print(f"Neo4j connection error: {str(e)}")
        return False
    finally:
        if 'driver' in locals():
            driver.close()

# Function to check if Qdrant is available
def check_qdrant_connection():
    """Check if Qdrant is available"""
    from qdrant_client import QdrantClient
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        response = client.get_collections()
        return True
    except Exception as e:
        print(f"Qdrant connection error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing database connections...")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Qdrant URL: {QDRANT_URL}")
    
    neo4j_available = check_neo4j_connection()
    print(f"Neo4j connection: {'SUCCESS' if neo4j_available else 'FAILED'}")
    
    qdrant_available = check_qdrant_connection()
    print(f"Qdrant connection: {'SUCCESS' if qdrant_available else 'FAILED'}")
    
    if not neo4j_available:
        print("\nNeo4j Connection Options:")
        print("1. Install Neo4j Desktop: https://neo4j.com/download/")
        print("2. Use Neo4j AuraDB (cloud): https://neo4j.com/cloud/aura/")
        
    if not qdrant_available:
        print("\nQdrant Connection Options:")
        print("1. Install Qdrant locally: https://qdrant.tech/documentation/quick-start/")
        print("2. Use Qdrant Cloud: https://cloud.qdrant.io/")
