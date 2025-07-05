"""
Qdrant Database Manager for GraphRAG Hybrid System
"""
import sys
import os
import uuid
import hashlib
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class QdrantManager:
    """
    Manages connections and operations with Qdrant vector database
    """
    
    def __init__(self, url=None, api_key=None, collection_name=None):
        """
        Initialize Qdrant connection
        
        Args:
            url (str): Qdrant URL
            api_key (str): Qdrant API key
            collection_name (str): Collection name
        """
        self.url = url or config.QDRANT_URL
        self.api_key = api_key or config.QDRANT_API_KEY
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.client = None
        self.connected = False
        self.vector_size = 384  # Default for all-MiniLM-L6-v2
        
        try:
            self.client = QdrantClient(url=self.url, api_key=self.api_key or None)
            # Test connection
            self.client.get_collections()
            self.connected = True
            logger.info(f"Successfully connected to Qdrant at {self.url}")
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            self.client = None
    
    def close(self):
        """Close the Qdrant connection"""
        self.client = None
        self.connected = False
        logger.info("Qdrant connection closed")
    
    def is_connected(self):
        """Check if connected to Qdrant"""
        return self.connected
    
    def string_to_uuid(self, string_id):
        """
        Convert a string ID to a UUID using a hash function
        
        Args:
            string_id (str): String ID
            
        Returns:
            str: UUID string
        """
        # Create a deterministic UUID from the string ID
        hash_object = hashlib.md5(string_id.encode())
        return str(uuid.UUID(hash_object.hexdigest()))
    
    def add_document(self, doc_id, embedding, metadata):
        """
        Add a document to Qdrant
        
        Args:
            doc_id (str): Document ID
            embedding (list): Document embedding
            metadata (dict): Document metadata
            
        Returns:
            bool: Success or failure
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Qdrant")
            return False
        
        try:
            # Convert string ID to UUID
            if isinstance(doc_id, str) and not doc_id.isdigit():
                uuid_id = self.string_to_uuid(doc_id)
                # Store the original ID in metadata for reference
                metadata["original_id"] = doc_id
            else:
                uuid_id = doc_id
                
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=uuid_id,
                        vector=embedding,
                        payload=metadata
                    )
                ]
            )
            logger.info(f"Added document {doc_id} to Qdrant with ID {uuid_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding document to Qdrant: {str(e)}")
            return False
    
    def add_purchase_order(self, po_number, embedding, po_data):
        """
        Add a purchase order to Qdrant
        
        Args:
            po_number (str): Purchase order number
            embedding (list): Document embedding
            po_data (dict): Purchase order data
            
        Returns:
            bool: Success or failure
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Qdrant")
            return False
        
        try:
            # Create metadata from PO data
            metadata = {
                "po_number": po_number,
                "date": po_data.get("date"),
                "total_value": po_data.get("total_value"),
                "project_name": po_data.get("project", {}).get("name"),
                "project_code": po_data.get("project", {}).get("code"),
                "supplier_name": po_data.get("supplier", {}).get("name"),
                "supplier_id": po_data.get("supplier", {}).get("id"),
                "item_count": len(po_data.get("items", [])),
                "payment_terms": po_data.get("payment_terms"),
                "delivery_terms": po_data.get("delivery_terms"),
                "type": "purchase_order"
            }
            
            return self.add_document(po_number, embedding, metadata)
        except Exception as e:
            logger.error(f"Error adding purchase order to Qdrant: {str(e)}")
            return False
    
    def search(self, query_embedding, limit=5, filter_condition=None):
        """
        Search for similar documents in Qdrant
        
        Args:
            query_embedding (list): Query embedding
            limit (int): Maximum number of results
            filter_condition (dict): Filter condition
            
        Returns:
            list: Search results
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Qdrant")
            return []
        
        try:
            # Create filter if provided
            filter_obj = None
            if filter_condition:
                filter_obj = models.Filter(**filter_condition)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_obj
            )
            
            return [
                {
                    "id": result.payload.get("original_id", result.id),
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {str(e)}")
            return []
    
    def search_purchase_orders(self, query_embedding, limit=5, project_name=None, supplier_name=None):
        """
        Search for similar purchase orders in Qdrant
        
        Args:
            query_embedding (list): Query embedding
            limit (int): Maximum number of results
            project_name (str): Filter by project name
            supplier_name (str): Filter by supplier name
            
        Returns:
            list: Search results
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Qdrant")
            return []
        
        try:
            # Create filter
            filter_conditions = [
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value="purchase_order")
                )
            ]
            
            if project_name:
                filter_conditions.append(
                    models.FieldCondition(
                        key="project_name",
                        match=models.MatchValue(value=project_name)
                    )
                )
            
            if supplier_name:
                filter_conditions.append(
                    models.FieldCondition(
                        key="supplier_name",
                        match=models.MatchValue(value=supplier_name)
                    )
                )
            
            filter_obj = models.Filter(
                must=filter_conditions
            )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=filter_obj
            )
            
            return [
                {
                    "po_number": result.payload.get("po_number"),
                    "project_name": result.payload.get("project_name"),
                    "supplier_name": result.payload.get("supplier_name"),
                    "date": result.payload.get("date"),
                    "total_value": result.payload.get("total_value"),
                    "score": result.score
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Error searching purchase orders in Qdrant: {str(e)}")
            return []
    
    def delete_document(self, doc_id):
        """
        Delete a document from Qdrant
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            bool: Success or failure
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Qdrant")
            return False
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[doc_id]
                )
            )
            logger.info(f"Deleted document {doc_id} from Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error deleting document from Qdrant: {str(e)}")
            return False

if __name__ == "__main__":
    # Test Qdrant connection
    qdrant_manager = QdrantManager()
    if qdrant_manager.is_connected():
        print("Successfully connected to Qdrant")
        qdrant_manager.close()
    else:
        print("Failed to connect to Qdrant")
