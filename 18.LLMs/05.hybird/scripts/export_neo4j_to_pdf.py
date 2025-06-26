"""
Neo4j Graph Exporter to PDF

This script exports Neo4j graph data to a PDF document with visualizations.
It processes nodes in batches to manage memory usage and combines all visualizations
into a single PDF file.
"""
import os
from datetime import datetime
from pyvis.network import Network
from neo4j import GraphDatabase
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
import tempfile

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "graphrag"
BATCH_SIZE = 25
OUTPUT_DIR = "neo4j_exports"
OUTPUT_PDF = f"neo4j_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Neo4jExporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
    
    def get_purchase_orders(self, limit=None, skip=0):
        """Retrieve purchase orders with related nodes"""
        query = """
        MATCH (po:PurchaseOrder)
        OPTIONAL MATCH (po)-[r]->(related)
        RETURN po, r, related
        ORDER BY po.po_number
        SKIP $skip
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit, skip=skip)
            return list(result)
    
    def get_total_purchase_orders(self):
        """Get total count of purchase orders"""
        query = "MATCH (po:PurchaseOrder) RETURN count(po) as count"
        with self.driver.session() as session:
            result = session.run(query)
            return result.single()["count"]

def create_visualization(data, batch_num):
    """Create a visualization of the graph data"""
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    
    # Track added nodes to avoid duplicates
    added_nodes = set()
    
    for record in data:
        # Add main PO node
        po = record["po"]
        po_id = f"po_{po.id}"
        if po_id not in added_nodes:
            net.add_node(po_id, label=po.get("po_number", "PO"), 
                        title=str(dict(po)), group="PurchaseOrder")
            added_nodes.add(po_id)
        
        # Add related node and edge if they exist
        if "r" in record and record["r"] and "related" in record and record["related"]:
            rel = record["r"]
            related = record["related"]
            related_id = f"{list(related.labels)[0]}_{related.id}"
            
            if related_id not in added_nodes:
                net.add_node(related_id, 
                           label=related.get("name", list(related.labels)[0]),
                           title=str(dict(related)),
                           group=list(related.labels)[0])
                added_nodes.add(related_id)
            
            net.add_edge(po_id, related_id, title=rel.type, label=rel.type)
    
    # Save visualization to a temporary file
    temp_file = os.path.join(OUTPUT_DIR, f"batch_{batch_num}.html")
    net.save_graph(temp_file)
    
    # Convert HTML to image using pyppeteer for better rendering
    return html_to_image(temp_file, f"batch_{batch_num}.png")

def html_to_image(html_file, output_file):
    """Convert HTML to image using pyppeteer"""
    try:
        from pyppeteer import launch
        import asyncio
        
        async def convert():
            browser = await launch()
            page = await browser.newPage()
            await page.setViewport({'width': 1200, 'height': 900})
            await page.goto(f'file://{os.path.abspath(html_file)}')
            await page.waitForSelector('canvas')
            output_path = os.path.join(OUTPUT_DIR, output_file)
            await page.screenshot({'path': output_path, 'fullPage': True})
            await browser.close()
            return output_path
            
        return asyncio.get_event_loop().run_until_complete(convert())
    except Exception as e:
        print(f"Warning: Could not use pyppeteer for rendering: {e}")
        # Fallback: Use a placeholder if pyppeteer fails
        return None

def create_pdf(image_files, output_pdf):
    """Combine images into a single PDF"""
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    
    for img_file in image_files:
        if img_file and os.path.exists(img_file):
            try:
                img = Image.open(img_file)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                # Scale image to fit page width
                new_width = width - 100  # Add margins
                new_height = new_width * aspect
                
                # If image is too tall, scale to fit page height
                if new_height > height - 100:
                    new_height = height - 100
                    new_width = new_height / aspect
                
                # Center the image
                x = (width - new_width) / 2
                y = (height - new_height) / 2
                
                c.drawImage(img_file, x, y, width=new_width, height=new_height)
                c.showPage()
            except Exception as e:
                print(f"Error adding {img_file} to PDF: {e}")
    
    c.save()

def main():
    print("Starting Neo4j to PDF export...")
    
    # Initialize Neo4j connection
    neo4j = Neo4jExporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Get total count
        total_pos = neo4j.get_total_purchase_orders()
        print(f"Found {total_pos} purchase orders in Neo4j")
        
        # Process in batches
        image_files = []
        for i in range(0, total_pos, BATCH_SIZE):
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(total_pos + BATCH_SIZE - 1)//BATCH_SIZE}...")
            
            # Get batch of data
            data = neo4j.get_purchase_orders(limit=BATCH_SIZE, skip=i)
            if not data:
                break
                
            # Create visualization
            img_file = create_visualization(data, i//BATCH_SIZE + 1)
            if img_file:
                image_files.append(img_file)
        
        # Create PDF
        if image_files:
            output_path = os.path.join(OUTPUT_DIR, OUTPUT_PDF)
            create_pdf(image_files, output_path)
            print(f"\nExport complete! PDF saved to: {os.path.abspath(output_path)}")
        else:
            print("No visualizations were created.")
            
    finally:
        neo4j.close()

if __name__ == "__main__":
    main()
