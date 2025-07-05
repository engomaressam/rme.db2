"""
Embedding Processor for GraphRAG Hybrid System
"""
import sys
import os
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class EmbeddingProcessor:
    """
    Generates embeddings for text using SentenceTransformers
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize embedding processor
        
        Args:
            model_name (str): Name of the SentenceTransformer model
        """
        self.model_name = model_name
        self.model = None
        
        try:
            # Check if CUDA is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load model
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
    
    def is_initialized(self):
        """Check if the model is initialized"""
        return self.model is not None
    
    def generate_embedding(self, text):
        """
        Generate embedding for text
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            list: Embedding vector
        """
        if not self.is_initialized():
            logger.error("Embedding model not initialized")
            return None
        
        try:
            # Generate embedding
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def generate_po_embedding(self, po_data):
        """
        Generate embedding for purchase order data
        
        Args:
            po_data (dict): Purchase order data
            
        Returns:
            list: Embedding vector
        """
        if not self.is_initialized():
            logger.error("Embedding model not initialized")
            return None
        
        try:
            # Create text representation of purchase order
            text = f"""
            Purchase Order: {po_data.get('po_number')}
            Date: {po_data.get('date')}
            Project: {po_data.get('project', {}).get('name')}
            Project Code: {po_data.get('project', {}).get('code')}
            Supplier: {po_data.get('supplier', {}).get('name')}
            Supplier ID: {po_data.get('supplier', {}).get('id')}
            Total Value: {po_data.get('total_value')}
            Payment Terms: {po_data.get('payment_terms')}
            Delivery Terms: {po_data.get('delivery_terms')}
            
            Items:
            """
            
            for item in po_data.get('items', []):
                text += f"""
                - {item.get('name')}: {item.get('quantity')} {item.get('unit')} at {item.get('value')}
                """
            
            # Generate embedding
            return self.generate_embedding(text)
        except Exception as e:
            logger.error(f"Error generating purchase order embedding: {str(e)}")
            return None
    
    def generate_query_embedding(self, query):
        """
        Generate embedding for query
        
        Args:
            query (str): Query text
            
        Returns:
            list: Embedding vector
        """
        return self.generate_embedding(query)

if __name__ == "__main__":
    # Test embedding processor
    processor = EmbeddingProcessor()
    if processor.is_initialized():
        # Test with a simple query
        query = "Purchase orders for Project X from Supplier Y"
        embedding = processor.generate_query_embedding(query)
        print(f"Generated embedding of length {len(embedding)}")
    else:
        print("Failed to initialize embedding processor")
