"""
GraphRAG Hybrid System for Purchase Order Analysis
"""
import sys
import os
import json
from loguru import logger
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.database.neo4j_manager import Neo4jManager
from src.database.qdrant_manager import QdrantManager
from src.processors.document_processor import DocumentProcessor
from src.processors.embedding_processor import EmbeddingProcessor

class GraphRAG:
    """
    GraphRAG Hybrid System for Purchase Order Analysis
    """
    
    def __init__(self, skip_llm=True, visualization_only=False):
        """
        Initialize GraphRAG system
        
        Args:
            skip_llm (bool): Skip LLM initialization
            visualization_only (bool): Run in visualization-only mode
        """
        self.skip_llm = skip_llm
        self.visualization_only = visualization_only
        
        # Initialize logger
        logger.info("Initializing GraphRAG Hybrid System")
        
        # Initialize components if not in visualization-only mode
        if not visualization_only:
            # Initialize Neo4j
            self.neo4j = Neo4jManager()
            if not self.neo4j.is_connected():
                logger.warning("Neo4j connection failed. Some features may not work.")
            
            # Initialize Qdrant
            self.qdrant = QdrantManager()
            if not self.qdrant.is_connected():
                logger.warning("Qdrant connection failed. Some features may not work.")
            
            # Initialize document processor
            self.doc_processor = DocumentProcessor()
            
            # Initialize embedding processor
            self.embedding_processor = EmbeddingProcessor()
            if not self.embedding_processor.is_initialized():
                logger.warning("Embedding processor initialization failed. Some features may not work.")
            
            # Initialize LLM if not skipped
            if not skip_llm:
                try:
                    # Import Ollama here to avoid errors if not installed
                    from langchain.llms import Ollama
                    self.llm = Ollama(model="mistral")
                    logger.info("Initialized Ollama LLM")
                except Exception as e:
                    logger.warning(f"Could not initialize Ollama LLM: {str(e)}")
                    self.llm = None
        
        # Initialize graph for visualization
        self.graph = nx.DiGraph()
        logger.info("GraphRAG initialization complete")
    
    def process_pdf(self, pdf_path, folder_name=None):
        """
        Process a PDF file and add to databases
        
        Args:
            pdf_path (str): Path to PDF file
            folder_name (str): Folder name to use as project name if available
            
        Returns:
            dict: Purchase order data
        """
        if self.visualization_only:
            logger.error("Cannot process PDFs in visualization-only mode")
            return None
        
        try:
            # Process PDF
            po_data = self.doc_processor.process_pdf_file(pdf_path, folder_name)
            if not po_data:
                logger.error(f"Failed to process PDF: {pdf_path}")
                return None
            
            # Generate embedding
            embedding = self.embedding_processor.generate_po_embedding(po_data)
            if not embedding:
                logger.error(f"Failed to generate embedding for PO: {po_data['po_number']}")
                return po_data
            
            # Add to Neo4j
            if self.neo4j.is_connected():
                success = self.neo4j.create_purchase_order(po_data)
                if success:
                    logger.info(f"Added PO {po_data['po_number']} to Neo4j")
                else:
                    logger.error(f"Failed to add PO {po_data['po_number']} to Neo4j")
            
            # Add to Qdrant
            if self.qdrant.is_connected():
                success = self.qdrant.add_purchase_order(po_data['po_number'], embedding, po_data)
                if success:
                    logger.info(f"Added PO {po_data['po_number']} to Qdrant")
                else:
                    logger.error(f"Failed to add PO {po_data['po_number']} to Qdrant")
            
            return po_data
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return None
    
    def process_directory(self, directory=None, recursive=True):
        """
        Process all PDF files in a directory
        
        Args:
            directory (str): Directory to process
            recursive (bool): Process subdirectories
            
        Returns:
            list: List of purchase order data
        """
        if self.visualization_only:
            logger.error("Cannot process PDFs in visualization-only mode")
            return []
        
        directory = directory or config.PDF_DIRECTORY
        logger.info(f"Processing directory: {directory}")
        
        po_data_list = self.doc_processor.process_directory(directory, recursive)
        
        # Add each PO to databases
        for po_data in po_data_list:
            # Generate embedding
            embedding = self.embedding_processor.generate_po_embedding(po_data)
            if not embedding:
                logger.error(f"Failed to generate embedding for PO: {po_data['po_number']}")
                continue
            
            # Add to Neo4j
            if self.neo4j.is_connected():
                success = self.neo4j.create_purchase_order(po_data)
                if success:
                    logger.info(f"Added PO {po_data['po_number']} to Neo4j")
                else:
                    logger.error(f"Failed to add PO {po_data['po_number']} to Neo4j")
            
            # Add to Qdrant
            if self.qdrant.is_connected():
                success = self.qdrant.add_purchase_order(po_data['po_number'], embedding, po_data)
                if success:
                    logger.info(f"Added PO {po_data['po_number']} to Qdrant")
                else:
                    logger.error(f"Failed to add PO {po_data['po_number']} to Qdrant")
        
        return po_data_list
    
    def search_similar_pos(self, query, limit=5, project_name=None, supplier_name=None):
        """
        Search for similar purchase orders
        
        Args:
            query (str): Query text
            limit (int): Maximum number of results
            project_name (str): Filter by project name
            supplier_name (str): Filter by supplier name
            
        Returns:
            list: Search results
        """
        if self.visualization_only:
            logger.error("Cannot search in visualization-only mode")
            return []
        
        if not self.qdrant.is_connected():
            logger.error("Qdrant not connected")
            return []
        
        # Generate query embedding
        embedding = self.embedding_processor.generate_query_embedding(query)
        if not embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search in Qdrant
        results = self.qdrant.search_purchase_orders(embedding, limit, project_name, supplier_name)
        logger.info(f"Found {len(results)} similar purchase orders")
        
        return results
    
    def get_project_pos(self, project_name):
        """
        Get all purchase orders for a project
        
        Args:
            project_name (str): Project name
            
        Returns:
            list: Purchase orders
        """
        if self.visualization_only:
            logger.error("Cannot query in visualization-only mode")
            return []
        
        if not self.neo4j.is_connected():
            logger.error("Neo4j not connected")
            return []
        
        results = self.neo4j.get_project_purchase_orders(project_name)
        logger.info(f"Found {len(results)} purchase orders for project {project_name}")
        
        return results
    
    def get_supplier_pos(self, supplier_name):
        """
        Get all purchase orders for a supplier
        
        Args:
            supplier_name (str): Supplier name
            
        Returns:
            list: Purchase orders
        """
        if self.visualization_only:
            logger.error("Cannot query in visualization-only mode")
            return []
        
        if not self.neo4j.is_connected():
            logger.error("Neo4j not connected")
            return []
        
        results = self.neo4j.get_supplier_purchase_orders(supplier_name)
        logger.info(f"Found {len(results)} purchase orders for supplier {supplier_name}")
        
        return results
    
    def get_project_suppliers(self, project_name):
        """
        Get all suppliers for a project
        
        Args:
            project_name (str): Project name
            
        Returns:
            list: Suppliers
        """
        if self.visualization_only:
            logger.error("Cannot query in visualization-only mode")
            return []
        
        if not self.neo4j.is_connected():
            logger.error("Neo4j not connected")
            return []
        
        results = self.neo4j.get_project_suppliers(project_name)
        logger.info(f"Found {len(results)} suppliers for project {project_name}")
        
        return results
    
    def get_supplier_projects(self, supplier_name):
        """
        Get all projects for a supplier
        
        Args:
            supplier_name (str): Supplier name
            
        Returns:
            list: Projects
        """
        if self.visualization_only:
            logger.error("Cannot query in visualization-only mode")
            return []
        
        if not self.neo4j.is_connected():
            logger.error("Neo4j not connected")
            return []
        
        results = self.neo4j.get_supplier_projects(supplier_name)
        logger.info(f"Found {len(results)} projects for supplier {supplier_name}")
        
        return results
    
    def build_graph_from_neo4j(self):
        """
        Build a graph from Neo4j data
        
        Returns:
            nx.DiGraph: NetworkX graph
        """
        if not self.neo4j.is_connected():
            logger.error("Neo4j not connected")
            return None
        
        try:
            # Clear existing graph
            self.graph = nx.DiGraph()
            
            # Get all projects
            projects = self.neo4j.run_query("MATCH (p:Project) RETURN p")
            for project in projects:
                project_data = project['p']
                self.graph.add_node(
                    project_data['name'],
                    type='Project',
                    code=project_data.get('code')
                )
            
            # Get all suppliers
            suppliers = self.neo4j.run_query("MATCH (s:Supplier) RETURN s")
            for supplier in suppliers:
                supplier_data = supplier['s']
                self.graph.add_node(
                    supplier_data['name'],
                    type='Supplier',
                    id=supplier_data.get('id'),
                    category=supplier_data.get('category')
                )
            
            # Get all purchase orders
            pos = self.neo4j.run_query("MATCH (po:PurchaseOrder) RETURN po")
            for po in pos:
                po_data = po['po']
                self.graph.add_node(
                    po_data['id'],
                    type='PurchaseOrder',
                    date=po_data.get('date'),
                    total_value=po_data.get('total_value')
                )
            
            # Get all items
            items = self.neo4j.run_query("MATCH (i:Item) RETURN i")
            for item in items:
                item_data = item['i']
                self.graph.add_node(
                    item_data['id'],
                    type='Item',
                    name=item_data.get('name'),
                    quantity=item_data.get('quantity'),
                    unit=item_data.get('unit'),
                    value=item_data.get('value')
                )
            
            # Get project-PO relationships
            project_pos = self.neo4j.run_query(
                "MATCH (p:Project)-[:HAS_PO]->(po:PurchaseOrder) RETURN p.name as project, po.id as po"
            )
            for rel in project_pos:
                self.graph.add_edge(rel['project'], rel['po'], type='HAS_PO')
            
            # Get supplier-PO relationships
            supplier_pos = self.neo4j.run_query(
                "MATCH (s:Supplier)-[:ISSUED_PO]->(po:PurchaseOrder) RETURN s.name as supplier, po.id as po"
            )
            for rel in supplier_pos:
                self.graph.add_edge(rel['supplier'], rel['po'], type='ISSUED_PO')
            
            # Get PO-item relationships
            po_items = self.neo4j.run_query(
                "MATCH (po:PurchaseOrder)-[:HAS_ITEM]->(i:Item) RETURN po.id as po, i.id as item"
            )
            for rel in po_items:
                self.graph.add_edge(rel['po'], rel['item'], type='HAS_ITEM')
            
            logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return self.graph
        except Exception as e:
            logger.error(f"Error building graph from Neo4j: {str(e)}")
            return None
    
    def build_graph_from_data(self, po_data_list):
        """
        Build a graph from purchase order data
        
        Args:
            po_data_list (list): List of purchase order data
            
        Returns:
            nx.DiGraph: NetworkX graph
        """
        try:
            # Clear existing graph
            self.graph = nx.DiGraph()
            
            for po_data in po_data_list:
                # Add project node
                if po_data.get('project') and po_data['project'].get('name'):
                    project_name = po_data['project']['name']
                    self.graph.add_node(
                        project_name,
                        type='Project',
                        code=po_data['project'].get('code')
                    )
                
                # Add supplier node
                if po_data.get('supplier') and po_data['supplier'].get('name'):
                    supplier_name = po_data['supplier']['name']
                    self.graph.add_node(
                        supplier_name,
                        type='Supplier',
                        id=po_data['supplier'].get('id'),
                        category=po_data['supplier'].get('category')
                    )
                
                # Add PO node
                po_number = po_data.get('po_number')
                self.graph.add_node(
                    po_number,
                    type='PurchaseOrder',
                    date=po_data.get('date'),
                    total_value=po_data.get('total_value')
                )
                
                # Add item nodes
                for i, item in enumerate(po_data.get('items', [])):
                    item_id = f"{po_number}-item-{i+1}"
                    self.graph.add_node(
                        item_id,
                        type='Item',
                        name=item.get('name'),
                        quantity=item.get('quantity'),
                        unit=item.get('unit'),
                        value=item.get('value')
                    )
                    
                    # Add PO-item relationship
                    self.graph.add_edge(po_number, item_id, type='HAS_ITEM')
                
                # Add project-PO relationship
                if po_data.get('project') and po_data['project'].get('name'):
                    self.graph.add_edge(project_name, po_number, type='HAS_PO')
                
                # Add supplier-PO relationship
                if po_data.get('supplier') and po_data['supplier'].get('name'):
                    self.graph.add_edge(supplier_name, po_number, type='ISSUED_PO')
            
            logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return self.graph
        except Exception as e:
            logger.error(f"Error building graph from data: {str(e)}")
            return None
    
    def load_graph_from_file(self, file_path):
        """
        Load a graph from a JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            nx.DiGraph: NetworkX graph
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create graph
            self.graph = nx.DiGraph()
            
            # Add nodes
            for node_id, node_data in data['nodes'].items():
                self.graph.add_node(node_id, **node_data)
            
            # Add edges
            for source, targets in data['edges'].items():
                for target, edge_data in targets.items():
                    self.graph.add_edge(source, target, **edge_data)
            
            logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return self.graph
        except Exception as e:
            logger.error(f"Error loading graph from file: {str(e)}")
            return None
    
    def save_graph_to_file(self, file_path=None):
        """
        Save the graph to a JSON file
        
        Args:
            file_path (str): Path to JSON file
            
        Returns:
            str: Path to saved file
        """
        if not file_path:
            # Create a file name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(config.OUTPUT_DIRECTORY, f"graph_{timestamp}.json")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert graph to dictionary
            data = {
                'nodes': {node: self.graph.nodes[node] for node in self.graph.nodes},
                'edges': {}
            }
            
            for source, target in self.graph.edges:
                if source not in data['edges']:
                    data['edges'][source] = {}
                data['edges'][source][target] = self.graph.edges[source, target]
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved graph to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving graph to file: {str(e)}")
            return None
    
    def visualize_graph(self, output_file=None, show=True):
        """
        Visualize the graph
        
        Args:
            output_file (str): Path to output file
            show (bool): Show the graph
            
        Returns:
            str: Path to saved file
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            logger.error("No graph to visualize")
            return None
        
        try:
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Define node colors by type
            node_colors = {
                'Project': '#4CAF50',  # Green
                'Supplier': '#2196F3',  # Blue
                'PurchaseOrder': '#FFC107',  # Amber
                'Item': '#9C27B0'  # Purple
            }
            
            # Get node positions using spring layout
            pos = nx.spring_layout(self.graph, seed=42)
            
            # Draw nodes by type
            for node_type, color in node_colors.items():
                nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == node_type]
                nx.draw_networkx_nodes(
                    self.graph, pos,
                    nodelist=nodes,
                    node_color=color,
                    node_size=500,
                    alpha=0.8,
                    label=node_type
                )
            
            # Draw edges
            nx.draw_networkx_edges(
                self.graph, pos,
                width=1.0,
                alpha=0.5,
                arrowsize=20
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                self.graph, pos,
                font_size=8,
                font_family='sans-serif'
            )
            
            # Add legend
            plt.legend()
            
            # Remove axis
            plt.axis('off')
            
            # Add title
            plt.title('Purchase Order Graph')
            
            # Save if output file provided
            if output_file:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                logger.info(f"Saved graph visualization to {output_file}")
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close()
            
            return output_file
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            return None
    
    def close(self):
        """Close all connections"""
        if not self.visualization_only:
            if hasattr(self, 'neo4j') and self.neo4j:
                self.neo4j.close()
            if hasattr(self, 'qdrant') and self.qdrant:
                self.qdrant.close()
        logger.info("Closed all connections")

if __name__ == "__main__":
    # Test GraphRAG
    graphrag = GraphRAG(skip_llm=True)
    
    # Process a directory of PDFs
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        if os.path.isdir(directory):
            po_data_list = graphrag.process_directory(directory)
            
            # Build graph from data
            graphrag.build_graph_from_data(po_data_list)
            
            # Visualize graph
            output_file = os.path.join(config.OUTPUT_DIRECTORY, "graph.png")
            graphrag.visualize_graph(output_file)
        else:
            print(f"Invalid directory: {directory}")
    else:
        print("No directory specified. Use: python graphrag.py <directory>")
