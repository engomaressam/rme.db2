"""
Neo4j Database Manager for GraphRAG Hybrid System
"""
import sys
import os
from loguru import logger
from neo4j import GraphDatabase

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class Neo4jManager:
    """
    Manages connections and operations with Neo4j database
    """
    
    def __init__(self, uri=None, user=None, password=None):
        """
        Initialize Neo4j connection
        
        Args:
            uri (str): Neo4j URI
            user (str): Neo4j username
            password (str): Neo4j password
        """
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.driver = None
        self.connected = False
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] == 1:
                    self.connected = True
                    logger.info(f"Successfully connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.driver = None
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.driver = None
            self.connected = False
            logger.info("Neo4j connection closed")
    
    def is_connected(self):
        """Check if connected to Neo4j"""
        return self.connected
    
    def run_query(self, query, parameters=None):
        """
        Run a Cypher query
        
        Args:
            query (str): Cypher query
            parameters (dict): Query parameters
            
        Returns:
            list: Query results
        """
        if not self.connected or not self.driver:
            logger.error("Not connected to Neo4j")
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {str(e)}")
            return None
    
    def create_purchase_order(self, po_data):
        """
        Create a purchase order node and its relationships
        
        Args:
            po_data (dict): Purchase order data
            
        Returns:
            bool: Success or failure
        """
        if not self.connected or not self.driver:
            logger.error("Not connected to Neo4j")
            return False
        
        try:
            with self.driver.session() as session:
                # Create PO node
                session.run(
                    """
                    MERGE (po:PurchaseOrder {id: $po_number})
                    SET po.date = $date,
                        po.total_value = $total_value,
                        po.payment_terms = $payment_terms,
                        po.delivery_terms = $delivery_terms
                    """,
                    {
                        "po_number": po_data.get("po_number"),
                        "date": po_data.get("date"),
                        "total_value": po_data.get("total_value"),
                        "payment_terms": po_data.get("payment_terms"),
                        "delivery_terms": po_data.get("delivery_terms")
                    }
                )
                
                # Create Project node and relationship
                if po_data.get("project") and po_data["project"].get("name"):
                    session.run(
                        """
                        MERGE (project:Project {name: $project_name})
                        SET project.code = $project_code
                        WITH project
                        MATCH (po:PurchaseOrder {id: $po_number})
                        MERGE (project)-[:HAS_PO]->(po)
                        """,
                        {
                            "project_name": po_data["project"]["name"],
                            "project_code": po_data["project"].get("code"),
                            "po_number": po_data.get("po_number")
                        }
                    )
                
                # Create Supplier node and relationship
                if po_data.get("supplier") and po_data["supplier"].get("name"):
                    session.run(
                        """
                        MERGE (supplier:Supplier {name: $supplier_name})
                        SET supplier.id = $supplier_id,
                            supplier.category = $supplier_category
                        WITH supplier
                        MATCH (po:PurchaseOrder {id: $po_number})
                        MERGE (supplier)-[:ISSUED_PO]->(po)
                        """,
                        {
                            "supplier_name": po_data["supplier"]["name"],
                            "supplier_id": po_data["supplier"].get("id"),
                            "supplier_category": po_data["supplier"].get("category"),
                            "po_number": po_data.get("po_number")
                        }
                    )
                
                # Create Item nodes and relationships
                for i, item in enumerate(po_data.get("items", [])):
                    item_id = f"{po_data.get('po_number')}-item-{i+1}"
                    session.run(
                        """
                        MERGE (item:Item {id: $item_id})
                        SET item.name = $name,
                            item.quantity = $quantity,
                            item.unit = $unit,
                            item.value = $value
                        WITH item
                        MATCH (po:PurchaseOrder {id: $po_number})
                        MERGE (po)-[:HAS_ITEM]->(item)
                        """,
                        {
                            "item_id": item_id,
                            "name": item.get("name"),
                            "quantity": item.get("quantity"),
                            "unit": item.get("unit"),
                            "value": item.get("value"),
                            "po_number": po_data.get("po_number")
                        }
                    )
                
                logger.info(f"Created purchase order {po_data.get('po_number')} in Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error creating purchase order in Neo4j: {str(e)}")
            return False
    
    def get_purchase_order(self, po_number):
        """
        Get a purchase order and its relationships
        
        Args:
            po_number (str): Purchase order number
            
        Returns:
            dict: Purchase order data
        """
        if not self.connected or not self.driver:
            logger.error("Not connected to Neo4j")
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (po:PurchaseOrder {id: $po_number})
                    OPTIONAL MATCH (project:Project)-[:HAS_PO]->(po)
                    OPTIONAL MATCH (supplier:Supplier)-[:ISSUED_PO]->(po)
                    OPTIONAL MATCH (po)-[:HAS_ITEM]->(item:Item)
                    RETURN po, project, supplier, collect(item) as items
                    """,
                    {"po_number": po_number}
                )
                
                record = result.single()
                if not record:
                    return None
                
                po_data = dict(record["po"])
                po_data["project"] = dict(record["project"]) if record["project"] else None
                po_data["supplier"] = dict(record["supplier"]) if record["supplier"] else None
                po_data["items"] = [dict(item) for item in record["items"]]
                
                return po_data
                
        except Exception as e:
            logger.error(f"Error retrieving purchase order from Neo4j: {str(e)}")
            return None
    
    def get_project_purchase_orders(self, project_name):
        """
        Get all purchase orders for a project
        
        Args:
            project_name (str): Project name
            
        Returns:
            list: Purchase orders
        """
        if not self.connected or not self.driver:
            logger.error("Not connected to Neo4j")
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (project:Project {name: $project_name})-[:HAS_PO]->(po:PurchaseOrder)
                    RETURN po
                    """,
                    {"project_name": project_name}
                )
                
                return [dict(record["po"]) for record in result]
                
        except Exception as e:
            logger.error(f"Error retrieving project purchase orders from Neo4j: {str(e)}")
            return None
    
    def get_supplier_purchase_orders(self, supplier_name):
        """
        Get all purchase orders for a supplier
        
        Args:
            supplier_name (str): Supplier name
            
        Returns:
            list: Purchase orders
        """
        if not self.connected or not self.driver:
            logger.error("Not connected to Neo4j")
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (supplier:Supplier {name: $supplier_name})-[:ISSUED_PO]->(po:PurchaseOrder)
                    RETURN po
                    """,
                    {"supplier_name": supplier_name}
                )
                
                return [dict(record["po"]) for record in result]
                
        except Exception as e:
            logger.error(f"Error retrieving supplier purchase orders from Neo4j: {str(e)}")
            return None
    
    def get_project_suppliers(self, project_name):
        """
        Get all suppliers for a project
        
        Args:
            project_name (str): Project name
            
        Returns:
            list: Suppliers
        """
        if not self.connected or not self.driver:
            logger.error("Not connected to Neo4j")
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (project:Project {name: $project_name})-[:HAS_PO]->(po:PurchaseOrder)<-[:ISSUED_PO]-(supplier:Supplier)
                    RETURN DISTINCT supplier
                    """,
                    {"project_name": project_name}
                )
                
                return [dict(record["supplier"]) for record in result]
                
        except Exception as e:
            logger.error(f"Error retrieving project suppliers from Neo4j: {str(e)}")
            return None
    
    def get_supplier_projects(self, supplier_name):
        """
        Get all projects for a supplier
        
        Args:
            supplier_name (str): Supplier name
            
        Returns:
            list: Projects
        """
        if not self.connected or not self.driver:
            logger.error("Not connected to Neo4j")
            return None
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (supplier:Supplier {name: $supplier_name})-[:ISSUED_PO]->(po:PurchaseOrder)<-[:HAS_PO]-(project:Project)
                    RETURN DISTINCT project
                    """,
                    {"supplier_name": supplier_name}
                )
                
                return [dict(record["project"]) for record in result]
                
        except Exception as e:
            logger.error(f"Error retrieving supplier projects from Neo4j: {str(e)}")
            return None

if __name__ == "__main__":
    # Test Neo4j connection
    neo4j_manager = Neo4jManager()
    if neo4j_manager.is_connected():
        print("Successfully connected to Neo4j")
        neo4j_manager.close()
    else:
        print("Failed to connect to Neo4j")
