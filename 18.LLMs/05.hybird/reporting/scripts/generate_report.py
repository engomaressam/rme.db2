import pandas as pd
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
import os
from datetime import datetime

# Import configuration
import sys
sys.path.append(r"c:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\18.llms\05.hybird")
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, QDRANT_URL, COLLECTION_NAME

class ProgressReport:
    def __init__(self):
        self.report_dir = "migration_reports"
        os.makedirs(self.report_dir, exist_ok=True)
        self.report_file = os.path.join(self.report_dir, f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
    def get_neo4j_stats(self):
        try:
            print(f"[DEBUG] Connecting to Neo4j at {NEO4J_URI}")
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
                print("[DEBUG] Connected to Neo4j, running queries...")
                with driver.session() as session:
                    # Get total documents
                    result = session.run("""
                        MATCH (po:PurchaseOrder)
                        RETURN count(po) as total_documents
                    """)
                    record = result.single() if result else None
                    total_docs = record["total_documents"] if record and "total_documents" in record else 0
                    print(f"[DEBUG] Found {total_docs} documents")
                    
                    # Get projects count
                    result = session.run("""
                        MATCH (p:Project)
                        RETURN count(p) as total_projects, 
                               collect(DISTINCT p.name) as project_names
                    """)
                    projects = result.single()
                    project_names = projects["project_names"] if projects and "project_names" in projects else []
                    
                    # Get document types
                    result = session.run("""
                        MATCH (po)
                        WHERE size(labels(po)) > 1
                        RETURN [label IN labels(po) WHERE label <> 'PurchaseOrder' | label] as doc_types
                    """)
                    doc_types = [t for r in result if "doc_types" in r for t in r["doc_types"]]
                    
                    return {
                        "total_documents": total_docs,
                        "total_projects": projects["total_projects"] if projects and "total_projects" in projects else 0,
                        "sample_projects": project_names[:5],  # Show first 5 projects
                        "document_types": list(set(doc_types))  # Unique document types
                    }
        except Exception as e:
            print(f"Error getting Neo4j stats: {e}")
            return {
                "total_documents": 0,
                "total_projects": 0,
                "sample_projects": [],
                "document_types": []
            }
    
    def get_qdrant_stats(self):
        try:
            print(f"[DEBUG] Connecting to Qdrant at {QDRANT_URL}")
            client = QdrantClient(QDRANT_URL)
            print(f"[DEBUG] Getting collection: {COLLECTION_NAME}")
            collection_info = client.get_collection(collection_name=COLLECTION_NAME)
            
            # Safely get values with defaults
            vectors_count = getattr(collection_info, 'vectors_count', 0)
            dimensions = 0
            distance_metric = "N/A"
            
            try:
                print("[DEBUG] Getting collection config...")
                if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                    if hasattr(collection_info.config.params, 'vectors'):
                        vectors = collection_info.config.params.vectors
                        if hasattr(vectors, 'size'):
                            dimensions = vectors.size
                        if hasattr(vectors, 'distance') and hasattr(vectors.distance, 'name'):
                            distance_metric = vectors.distance.name
            except Exception as e:
                print(f"[WARNING] Could not get all Qdrant metrics: {e}")
            
            stats = {
                "total_vectors": vectors_count,
                "dimensions": dimensions,
                "distance_metric": distance_metric
            }
            print(f"[DEBUG] Qdrant stats: {stats}")
            return stats
        except Exception as e:
            print(f"Error getting Qdrant stats: {e}")
            return {
                "total_vectors": 0,
                "dimensions": 0,
                "distance_metric": "N/A"
            }
    
    def generate_plots(self, stats):
        try:
            # Create a simple bar chart of document types if we have them
            document_types = stats.get('document_types', [])
            if document_types and len(document_types) > 0:
                plt.figure(figsize=(10, 6))
                doc_type_counts = {}
                for doc_type in document_types:
                    if doc_type:  # Skip any None or empty types
                        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
                
                if doc_type_counts:  # Only plot if we have valid document types
                    plt.bar(doc_type_counts.keys(), doc_type_counts.values())
                    plt.title('Document Types Distribution')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plot_path = os.path.join(self.report_dir, 'document_types.png')
                    plt.savefig(plot_path)
                    plt.close()
                    return plot_path
            
            # If we get here, create a simple status message instead of a plot
            plt.figure(figsize=(10, 2))
            plt.text(0.5, 0.5, 'No document types data available yet', 
                     ha='center', va='center', fontsize=12)
            plt.axis('off')
            plot_path = os.path.join(self.report_dir, 'document_types.png')
            plt.savefig(plot_path)
            plt.close()
            return plot_path
        except Exception as e:
            print(f"Error generating plots: {e}")
        return None
    
    def generate_html_report(self):
        print("Gathering migration statistics...")
        try:
            print("[DEBUG] Getting Neo4j stats...")
            neo4j_stats = self.get_neo4j_stats()
            print("[DEBUG] Getting Qdrant stats...")
            qdrant_stats = self.get_qdrant_stats()
            
            stats = {
                'neo4j': neo4j_stats or {
                    'total_documents': 0,
                    'total_projects': 0,
                    'sample_projects': [],
                    'document_types': []
                },
                'qdrant': qdrant_stats or {
                    'total_vectors': 0,
                    'dimensions': 0,
                    'distance_metric': 'N/A'
                }
            }
            print(f"[DEBUG] Stats collected: {stats}")
        except Exception as e:
            print(f"[ERROR] Error collecting stats: {e}")
            stats = {
                'neo4j': {
                    'total_documents': 0,
                    'total_projects': 0,
                    'sample_projects': [],
                    'document_types': []
                },
                'qdrant': {
                    'total_vectors': 0,
                    'dimensions': 0,
                    'distance_metric': 'N/A'
                }
            }
        
        print("Generating visualizations...")
        plot_path = self.generate_plots(stats['neo4j'])
        
        # Generate HTML
        html = f"""
        <html>
        <head>
            <title>Document Migration Progress Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
                .metric {{ margin-bottom: 20px; padding: 15px; background: #f4f4f4; border-radius: 5px; }}
                h1, h2, h3 {{ color: #333; }}
                .grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .card {{ flex: 1; min-width: 200px; background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .success {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
            </style>
        </head>
        <body>
            <h1>Document Migration Progress Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Migration Overview</h2>
            <div class="grid">
                <div class="card">
                    <h3>📄 Documents Processed</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #2c3e50;">
                        {stats['neo4j'].get('total_documents', 0) or 0:,}
                    </p>
                </div>
                <div class="card">
                    <h3>🏗️ Projects Identified</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #27ae60;">
                        {stats['neo4j'].get('total_projects', 0) or 0:,}
                    </p>
                </div>
                <div class="card">
                    <h3>📊 Document Types</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #8e44ad;">
                        {len(stats['neo4j'].get('document_types', []))}
                    </p>
                </div>
            </div>
            """
            
        # Add Vector Database Status
        html += f"""
        <h2>Vector Database Status</h2>
        <div class="metric">
                <p>Vectors Stored: <strong>{(stats['qdrant'].get('total_vectors') or 0):,}</strong></p>
                <p>Vector Dimensions: <strong>{stats['qdrant'].get('dimensions', 'N/A')}</strong></p>
                <p>Similarity Metric: <strong>{stats['qdrant'].get('distance_metric', 'N/A')}</strong></p>
        </div>
        """
            
        # Add Sample Projects if available
        if stats['neo4j']['sample_projects']:
            html += """
            <h2>Sample Projects</h2>
            <ul>
            """
            html += '\n'.join(f'<li>{project}</li>' for project in stats['neo4j']['sample_projects'])
            html += """
            </ul>
            """
            
        # Add Document Types if available
        if stats['neo4j']['document_types']:
            html += """
            <h2>Document Types</h2>
            <ul>
            """
            html += '\n'.join(f'<li>{doc_type}</li>' for doc_type in stats['neo4j']['document_types'])
            html += """
            </ul>
            """
            
        # Add plot if available
        if plot_path:
            html += f"""
            <h2>Document Types Distribution</h2>
            <img src="{os.path.basename(plot_path)}" alt="Document Types Distribution" style="max-width: 100%;">
            """
            
        # Add status message
        if stats['neo4j']['total_documents'] > 0 or stats['qdrant']['total_vectors'] > 0:
            status_class = "success"
            status_msg = "✅ Migration is in progress and data is being processed successfully."
        else:
            status_class = "warning"
            status_msg = "⚠️ No data found. Please check if the migration script is running and has processed any data."
            
        html += f"""
        <div class="metric {status_class}" style="margin-top: 30px;">
            <h2>Status</h2>
            <p>{status_msg}</p>
        </div>
        </body>
        </html>
        """
        
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(html)
            
        return self.report_file

if __name__ == "__main__":
    print("Generating migration progress report...")
    reporter = ProgressReport()
    try:
        report_path = reporter.generate_html_report()
        print(f"[SUCCESS] Report generated successfully: {os.path.abspath(report_path)}")
        print(f"[INFO] You can open this file in your web browser to view the report.")
    except Exception as e:
        print(f"[ERROR] Error generating report: {e}")
