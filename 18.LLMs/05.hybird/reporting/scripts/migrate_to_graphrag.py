"""
MySQL to GraphRAG Data Migration Script

This script migrates data from the MySQL po.pdfs table to the GraphRAG system (Neo4j + Qdrant).
It uses the existing extracted text from the database to create embeddings and store them in Qdrant,
while also creating appropriate nodes and relationships in Neo4j.
"""

import os
import sys
import json
import logging
import importlib
from typing import Dict, List, Any, Optional
from pathlib import Path
import mysql.connector
from mysql.connector import Error
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

# Import configuration
CONFIG_PATH = r"c:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\18.llms\05.hybird\config.py"
spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
sys.modules["config"] = config
spec.loader.exec_module(config)

# Import configuration variables
NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD
QDRANT_URL = config.QDRANT_URL
EMBEDDING_MODEL = config.EMBEDDING_MODEL
COLLECTION_NAME = config.COLLECTION_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MySQLToGraphRAGMigrator:
    """Handles the migration of data from MySQL to GraphRAG system."""
    
    def __init__(self, batch_size: int = 100):
        """Initialize the migrator with database connections."""
        self.batch_size = batch_size
        self.mysql_conn = None
        self.neo4j_driver = None
        self.qdrant_client = None
        self.embedding_model = None
        
        # Initialize connections
        self._connect_to_databases()
        self._load_embedding_model()
    
    def _connect_to_databases(self):
        """Establish connections to MySQL, Neo4j, and Qdrant."""
        # MySQL Connection
        try:
            self.mysql_conn = mysql.connector.connect(
                host='10.10.11.242',
                user='omar2',
                password='Omar_54321',
                database='RME_TEST'
            )
            logger.info("Successfully connected to MySQL database")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise
        
        # Neo4j Connection
        try:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            # Test connection
            with self.neo4j_driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            self.close_connections()
            raise
        
        # Qdrant Connection
        try:
            self.qdrant_client = QdrantClient(QDRANT_URL)
            # Test connection
            self.qdrant_client.get_collections()
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            self.close_connections()
            raise
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for generating embeddings."""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=device
            )
            logger.info(f"Loaded embedding model: all-MiniLM-L6-v2 on {device}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.close_connections()
            raise
    
    def close_connections(self):
        """Close all database connections."""
        if hasattr(self, 'mysql_conn') and self.mysql_conn and self.mysql_conn.is_connected():
            self.mysql_conn.close()
            logger.info("MySQL connection closed")
        
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed")
    
    def create_qdrant_collection(self, collection_name: str, vector_size: int = 384):
        """Create a Qdrant collection if it doesn't exist."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {collection_name}")
                
            return True
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            return False
    
    def migrate_data(self):
        """Main method to migrate data from MySQL to GraphRAG."""
        try:
            # Create Qdrant collection
            if not self.create_qdrant_collection(COLLECTION_NAME):
                raise Exception("Failed to create Qdrant collection")
            
            # Get total count of records to process
            cursor = self.mysql_conn.cursor(dictionary=True)
            cursor.execute("SELECT COUNT(*) as count FROM `po.pdfs` WHERE extracted_text IS NOT NULL")
            total_records = cursor.fetchone()['count']
            
            if total_records == 0:
                logger.warning("No records with extracted text found in the database")
                return
            
            logger.info(f"Starting migration of {total_records} purchase orders")
            
            # Process in batches
            offset = 0
            processed_count = 0
            
            with tqdm(total=total_records, desc="Migrating purchase orders") as pbar:
                while True:
                    # Fetch a batch of records
                    cursor.execute("""
                        SELECT 
                            id, 
                            pdf_path,
                            pdf_filename,
                            pdf_hash,
                            extracted_text,
                            project_name,
                            document_type
                        FROM `po.pdfs` 
                        WHERE extracted_text IS NOT NULL
                        LIMIT %s OFFSET %s
                    """, (self.batch_size, offset))
                    
                    batch = cursor.fetchall()
                    if not batch:
                        break
                    
                    # Process batch
                    self._process_batch(batch, COLLECTION_NAME)
                    
                    # Update counters
                    processed_count += len(batch)
                    offset += self.batch_size
                    pbar.update(len(batch))
            
            logger.info(f"Successfully migrated {processed_count} purchase orders")
            
        except Exception as e:
            logger.error(f"Error during migration: {e}", exc_info=True)
            raise
        finally:
            cursor.close()
    
    def _process_batch(self, batch: List[Dict[str, Any]], collection_name: str):
        """Process a batch of records and store in Neo4j and Qdrant."""
        try:
            # Prepare data for Qdrant
            points = []
            
            for record in batch:
                # Generate embedding from extracted text
                text = record['extracted_text']
                if not text or len(text.strip()) == 0:
                    continue
                
                embedding = self.embedding_model.encode(text, convert_to_tensor=False).tolist()
                
                # Create point for Qdrant
                point = PointStruct(
                    id=record['id'],
                    vector=embedding,
                    payload={
                        'id': record['id'],
                        'pdf_path': record['pdf_path'],
                        'pdf_filename': record['pdf_filename'],
                        'pdf_hash': record['pdf_hash'],
                        'project_name': record['project_name'],
                        'document_type': record['document_type'],
                        'text': text[:1000] + '...' if len(text) > 1000 else text  # Store first 1000 chars
                    }
                )
                points.append(point)
                
                # Create/update node in Neo4j
                self._update_neo4j(record)
            
            # Upsert points to Qdrant
            if points:
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            raise
    
    def _update_neo4j(self, record: Dict[str, Any]):
        """Create or update a node in Neo4j."""
        try:
            with self.neo4j_driver.session() as session:
                # Create or merge PurchaseOrder node
                query = """
                MERGE (po:PurchaseOrder {id: $id})
                ON CREATE SET 
                    po.pdf_path = $pdf_path,
                    po.pdf_filename = $pdf_filename,
                    po.pdf_hash = $pdf_hash,
                    po.created_at = datetime()
                ON MATCH SET
                    po.updated_at = datetime()
                RETURN po
                """
                session.run(query, {
                    'id': record['id'],
                    'pdf_path': record['pdf_path'],
                    'pdf_filename': record['pdf_filename'],
                    'pdf_hash': record['pdf_hash']
                })
                
                # If project_name exists, create Project node and relationship
                if record.get('project_name'):
                    query = """
                    MATCH (po:PurchaseOrder {id: $po_id})
                    MERGE (pr:Project {name: $project_name})
                    MERGE (po)-[:BELONGS_TO]->(pr)
                    """
                    session.run(query, {
                        'po_id': record['id'],
                        'project_name': record['project_name']
                    })
                
                # If document_type exists, add as label
                if record.get('document_type'):
                    query = """
                    MATCH (po:PurchaseOrder {id: $id})
                    SET po :%s
                    """ % record['document_type'].replace(' ', '_').upper()
                    session.run(query, {'id': record['id']})
                    
        except Exception as e:
            logger.error(f"Error updating Neo4j: {e}")
            raise

def main():
    """Main function to run the migration."""
    migrator = None
    try:
        migrator = MySQLToGraphRAGMigrator(batch_size=50)
        migrator.migrate_data()
        logger.info("Migration completed successfully!")
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if migrator:
            migrator.close_connections()

if __name__ == "__main__":
    main()
