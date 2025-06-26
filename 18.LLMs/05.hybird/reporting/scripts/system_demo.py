import os
import datetime 
from typing import Dict, Any
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uuid
import pkg_resources
try:
    print(f"Qdrant client version: {pkg_resources.get_distribution('qdrant-client').version}")
except pkg_resources.DistributionNotFound:
    print("Qdrant client version: Not found (qdrant-client package not found by pkg_resources)")
except Exception as e:
    print(f"Qdrant client version: Error getting version - {e}")

import qdrant_client # Keep the original import for the rest of the script

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password") 
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333") 
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "purchase_orders")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2") 
REPORT_DIR = "system_reports" 

class SystemDemo:
    def __init__(self):
        print("Initializing SystemDemo...")
        self.neo4j_driver = None
        self.qdrant_client = None
        self.embedding_model = None

        try:
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.neo4j_driver.verify_connectivity()
            print("Neo4j connection successful.")
        except Exception as e:
            print(f"[ERROR] Neo4j connection failed: {e}")
            self.neo4j_driver = None

        try:
            self.qdrant_client = QdrantClient(url=QDRANT_URL)
            print("Qdrant client initialized.")
        except Exception as e:
            print(f"[ERROR] Qdrant client initialization failed: {e}")
            self.qdrant_client = None

        try:
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
            self.embedding_model = None
        
        self.report_dir = REPORT_DIR
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
            print(f"Report directory created: {self.report_dir}")

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()
            print("Neo4j connection closed.")

    def get_system_stats(self) -> Dict[str, Any]:
        stats = {'node_counts': [], 'relationship_counts': []}
        
        if not self.neo4j_driver:
            stats['neo4j_error'] = "Neo4j driver not initialized."
        else:
            try:
                with self.neo4j_driver.session() as session:
                    node_counts_result = session.run("""
                        MATCH (n)
                        RETURN labels(n)[0] as label, count(*) as count
                        ORDER BY count DESC
                    """)
                    stats['node_counts'] = [dict(record) for record in node_counts_result]
                    
                    rel_counts_result = session.run("""
                        MATCH ()-[r]->()
                        RETURN type(r) as type, count(*) as count
                        ORDER BY count DESC
                    """)
                    stats['relationship_counts'] = [dict(record) for record in rel_counts_result]
            except Exception as e:
                print(f"[ERROR] Could not get Neo4j stats: {e}")
                stats['neo4j_error'] = str(e)
        
        stats.setdefault('node_counts', [])
        stats.setdefault('relationship_counts', [])

        if not self.qdrant_client:
            stats['qdrant_stats'] = {'error': "Qdrant client not initialized.", 'vectors_count': 0, 'dimensions': 'N/A', 'distance_metric': 'N/A', 'collection_name': COLLECTION_NAME}
        else:
            try:
                self.qdrant_client.get_collection(collection_name=COLLECTION_NAME) 
                collection_info = self.qdrant_client.get_collection(collection_name=COLLECTION_NAME)
                stats['qdrant_stats'] = {
                    'vectors_count': collection_info.vectors_count if collection_info.vectors_count is not None else 0,
                    'dimensions': collection_info.config.params.vectors.size if collection_info.config and collection_info.config.params and collection_info.config.params.vectors else 'N/A',
                    'distance_metric': str(collection_info.config.params.vectors.distance).upper() if collection_info.config and collection_info.config.params and collection_info.config.params.vectors else 'N/A',
                    'collection_name': COLLECTION_NAME
                }
            except Exception as e: 
                print(f"[ERROR] Could not get Qdrant stats for collection '{COLLECTION_NAME}': {e}")
                stats['qdrant_stats'] = {'error': str(e), 'vectors_count': 0, 'dimensions': 'N/A', 'distance_metric': 'N/A', 'collection_name': COLLECTION_NAME}
        return stats

    def run_sample_queries(self) -> Dict[str, Any]:
        results = {}
        default_query_text = "construction materials delivery" 
        
        results['top_suppliers'] = []
        results['recent_purchase_orders'] = []
        results['top_projects'] = []
        results['semantic_search'] = {'query': default_query_text, 'results': [], 'error': None}
        results['nearest_neighbors'] = {'node_id': 'N/A', 'results': [], 'error': None}

        if not self.neo4j_driver:
            results['neo4j_query_error'] = "Neo4j driver not initialized."
        else:
            try:
                with self.neo4j_driver.session() as session:
                    top_suppliers_q = """
                        MATCH (s:Supplier)<-[:SUPPLIED_BY]-(po:PurchaseOrder) 
                        RETURN s.name as supplier, count(po) as po_count, sum(po.amount) as total_amount
                        ORDER BY total_amount DESC LIMIT 10
                    """
                    results['top_suppliers'] = [dict(record) for record in session.run(top_suppliers_q)]
                    
                    recent_po_q = """
                        MATCH (po:PurchaseOrder)<-[:HAS_PO]-(p:Project)
                        OPTIONAL MATCH (po)-[:CONTAINS_ITEM]->(i:Item)
                        WITH po, p, count(i) as item_count
                        RETURN po.id as po_number, po.date as date, po.amount as amount, p.name as project, item_count
                        ORDER BY po.date DESC LIMIT 10
                    """
                    results['recent_purchase_orders'] = [dict(record) for record in session.run(recent_po_q)]

                    top_projects_q = """
                        MATCH (p:Project)-[:HAS_PO]->(po:PurchaseOrder)
                        RETURN p.name as project, count(po) as po_count, sum(po.amount) as total_amount
                        ORDER BY po_count DESC LIMIT 10
                    """
                    results['top_projects'] = [dict(record) for record in session.run(top_projects_q)]
            except Exception as e:
                print(f"[ERROR] Neo4j query failed during sample queries: {e}")
                results['neo4j_query_error'] = str(e)

        if not self.qdrant_client or not self.embedding_model:
            error_msg_parts = []
            if not self.qdrant_client: error_msg_parts.append("Qdrant client not initialized.")
            if not self.embedding_model: error_msg_parts.append("Embedding model not loaded.")
            qdrant_error = " ".join(error_msg_parts)
            results['semantic_search']['error'] = qdrant_error
            results['nearest_neighbors']['error'] = qdrant_error
        else:
            try:
                query_embedding = self.embedding_model.encode(default_query_text).tolist()
                search_hits = self.qdrant_client.query_points(
                    COLLECTION_NAME, # Positional arg for collection name
                    query_embedding, # Positional arg for query vector
                    limit=5,
                    with_payload=True
                    # collection_name=COLLECTION_NAME, # Keep for clarity if needed, but might be redundant if positional works
                ).points
                results['semantic_search']['results'] = [
                    {
                        'id': hit.id,
                        'score': hit.score,
                        'labels': hit.payload.get('labels', []) if hit.payload else [],
                        'properties': hit.payload if hit.payload else {}
                    } for hit in search_hits
                ]
            except Exception as e:
                print(f"[ERROR] Qdrant semantic search failed: {e}")
                results['semantic_search']['error'] = str(e)
            
            if results.get('recent_purchase_orders'): 
                sample_po_neo4j_id = results['recent_purchase_orders'][0].get('po_number')
                results['nearest_neighbors']['node_id'] = sample_po_neo4j_id if sample_po_neo4j_id else 'N/A'
                
                if sample_po_neo4j_id:
                    try:
                        qdrant_point_id_to_lookup_str = sample_po_neo4j_id
                        # Convert string ID to a UUID string for Qdrant
                        qdrant_id_as_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, qdrant_point_id_to_lookup_str))
                        retrieved_points_response = self.qdrant_client.retrieve(
                            collection_name=COLLECTION_NAME,
                            ids=[qdrant_id_as_uuid],
                            with_vectors=True
                        )
                        
                        if retrieved_points_response and isinstance(retrieved_points_response, list) and len(retrieved_points_response) > 0 and hasattr(retrieved_points_response[0], 'vector') and retrieved_points_response[0].vector:
                            po_vector = retrieved_points_response[0].vector
                            neighbor_hits = self.qdrant_client.query_points(
                                COLLECTION_NAME, # Positional arg for collection name
                                po_vector,       # Positional arg for query vector
                                limit=6, 
                                with_payload=True
                                # collection_name=COLLECTION_NAME, # Keep for clarity if needed
                            ).points
                            results['nearest_neighbors']['results'] = [
                                {
                                    'id': hit.id,
                                    'score': hit.score,
                                    'labels': hit.payload.get('labels', []) if hit.payload else [],
                                    'properties': hit.payload if hit.payload else {}
                                } for hit in neighbor_hits if str(hit.id) != qdrant_id_as_uuid 
                            ][:5]
                        else:
                            warn_msg = f"Could not retrieve vector for Qdrant point ID: {qdrant_id_as_uuid} (derived from Neo4j ID: {sample_po_neo4j_id})"
                            print(f"[WARNING] {warn_msg}")
                            results['nearest_neighbors']['error'] = warn_msg
                    except Exception as e:
                        warn_msg = f"Nearest neighbor search failed for Neo4j ID {sample_po_neo4j_id} (Qdrant ID {qdrant_id_as_uuid if 'qdrant_id_as_uuid' in locals() else 'unknown'}): {type(e).__name__} - {e}"
                        print(f"[WARNING] {warn_msg}")
                        results['nearest_neighbors']['error'] = warn_msg
                else: 
                    results['nearest_neighbors']['error'] = 'No sample PO ID found in recent_purchase_orders for neighbor search.'
            else: 
                 results['nearest_neighbors']['error'] = 'No recent_purchase_orders available for neighbor search.'
        return results

    def generate_reports(self):
        print("\nGenerating reports...")
        system_stats = self.get_system_stats()
        query_results = self.run_sample_queries()

        # Dynamically import report classes to avoid circular dependencies at module load time
        # and to handle cases where report files might not exist yet.
        try:
            from neo4j_report import Neo4jReport
            if self.neo4j_driver:
                neo4j_reporter = Neo4jReport(self.report_dir)
                print(f"\n[DEBUG] System stats for Neo4j report: {system_stats}\n")
                neo4j_report_path = neo4j_reporter.generate(system_stats, query_results)
                if neo4j_report_path:
                    print(f"Neo4j report generated: {neo4j_report_path}")
                else:
                    print("[ERROR] Failed to generate Neo4j report.")
            else:
                print("[INFO] Skipping Neo4j report generation as Neo4j driver is not available.")
        except ImportError:
            print("[ERROR] neo4j_report.py not found or Neo4jReport class cannot be imported.")
        except Exception as e:
            print(f"[ERROR] Error during Neo4j report generation: {e}")


        try:
            from qdrant_report import QdrantReport
            if self.qdrant_client:
                qdrant_reporter = QdrantReport(self.report_dir)
                print(f"\n[DEBUG] System stats for Qdrant report: {system_stats}\n")
                qdrant_report_path = qdrant_reporter.generate(system_stats, query_results)
                if qdrant_report_path:
                    print(f"Qdrant report generated: {qdrant_report_path}")
                else:
                    print("[ERROR] Failed to generate Qdrant report.")
            else:
                print("[INFO] Skipping Qdrant report generation as Qdrant client is not available.")
        except ImportError:
            print("[ERROR] qdrant_report.py not found or QdrantReport class cannot be imported.")
        except Exception as e:
            print(f"[ERROR] Error during Qdrant report generation: {e}")


if __name__ == "__main__":
    print("Starting System Demo...")
    demo = SystemDemo()
    
    # Check if essential components are initialized
    if not demo.neo4j_driver and not demo.qdrant_client:
        print("\n[CRITICAL] Both Neo4j and Qdrant connections failed. Cannot generate meaningful reports.")
    elif not demo.embedding_model:
        print("\n[CRITICAL] Embedding model failed to load. Qdrant semantic features will be affected.")
        # Decide if you still want to generate reports (e.g., Neo4j part might still work)
        demo.generate_reports() # Or conditionally skip Qdrant parts
    else:
        demo.generate_reports()
        
    demo.close()
    print("\nSystem demo finished.")

