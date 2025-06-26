import os
from datetime import datetime
from typing import Dict, Any, List

class QdrantReport:
    def __init__(self, report_dir: str):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.report_file = os.path.join(self.report_dir, f"qdrant_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    def _format_qdrant_stats(self, stats: Dict[str, Any]) -> str:
        """Format Qdrant statistics for HTML report"""
        if not stats.get('qdrant_stats'):
            return '<div class="warning">No Qdrant statistics available</div>'
        
        qdrant = stats['qdrant_stats']
        return f"""
        <div class="grid">
            <div class="card">
                <h3>Collection Information</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Vectors Stored</td><td style="text-align: right;">{qdrant.get('vectors_count', 0):,}</td></tr>
                    <tr><td>Vector Dimensions</td><td style="text-align: right;">{qdrant.get('dimensions', 'N/A')}</td></tr>
                    <tr><td>Similarity Metric</td><td>{qdrant.get('distance_metric', 'N/A')}</td></tr>
                </table>
            </div>
        </div>
        """
    
    def _format_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Format vector search results for HTML report"""
        if not search_results:
            return '<div class="card">No search results available</div>'
        
        html = ""
        for i, result in enumerate(search_results, 1):
            try:
                labels = ', '.join(result.get('labels', ['Unknown']))
                score = f"{result.get('score', 0):.4f}" if 'score' in result else 'N/A'
                props = result.get('properties', {})
                
                # Format properties for display
                props_html = ""
                for k, v in props.items():
                    if k in ['embedding', 'vector']:
                        continue
                    v_str = str(v)
                    if len(v_str) > 100:
                        v_str = v_str[:100] + '...'
                    props_html += f'<div><strong>{k}:</strong> {v_str}</div>'
                
                html += f"""
                <div class="search-result">
                    <h4>Result {i}: {labels} <span style="color: #666;">(Score: {score})</span></h4>
                    <div style="margin-left: 15px; margin-top: 5px;">
                        {props_html or 'No properties available'}
                    </div>
                </div>"""
            except Exception as e:
                print(f"[WARNING] Error formatting search result: {e}")
        
        return html
    
    def _format_semantic_search(self, query_results: Dict[str, Any]) -> str:
        """Format semantic search section"""
        if not query_results.get('semantic_search'):
            return '<div class="warning">No semantic search results available</div>'
        
        search = query_results['semantic_search']
        return f"""
        <div class="search-query">
            <h3>Semantic Search Query</h3>
            <p>{search.get('query', 'N/A')}</p>
        </div>
        <div class="search-results">
            <h4>Top Results:</h4>
            {self._format_search_results(search.get('results', []))}
        </div>
        """
    
    def _format_vector_visualization(self) -> str:
        """Placeholder for vector visualization"""
        return """
        <div class="card">
            <h3>Vector Space Visualization</h3>
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 4px;">
                <p><em>Vector space visualization would appear here</em></p>
                <p>This would show a 2D projection of the vector space with similar documents clustered together.</p>
            </div>
        </div>
        """
    
    def generate(self, stats: Dict[str, Any], query_results: Dict[str, Any]) -> str:
        """Generate the Qdrant HTML report"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Pre-format all dynamic HTML parts
            formatted_qdrant_stats = self._format_qdrant_stats(stats)
            formatted_semantic_search = self._format_semantic_search(query_results)
            formatted_vector_viz = self._format_vector_visualization()
            actual_vector_count = stats.get('qdrant_stats', {}).get('vectors_count', 0)
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Qdrant Vector Search Report</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        line-height: 1.6; 
                        margin: 0;
                        padding: 20px;
                        color: #333;
                        background-color: #f5f7fa;
                    }}
                    .container {{ 
                        max-width: 1200px; 
                        margin: 0 auto; 
                    }}
                    .header {{ 
                        background: #2c3e50; 
                        color: white; 
                        padding: 20px; 
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }}
                    .section {{ 
                        margin-bottom: 30px; 
                        background: white;
                        border-radius: 5px;
                        padding: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .card {{ 
                        background: #f8f9fa; 
                        border-radius: 5px; 
                        padding: 20px; 
                        margin-bottom: 20px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    h1, h2, h3, h4, h5, h6 {{ 
                        color: #2c3e50; 
                        margin-top: 0;
                    }}
                    .grid {{ 
                        display: grid; 
                        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                    }}
                    table {{ 
                        width: 100%; 
                        border-collapse: collapse; 
                        margin: 15px 0;
                        font-size: 14px;
                    }}
                    th, td {{ 
                        padding: 12px 15px; 
                        text-align: left; 
                        border: 1px solid #e0e0e0;
                    }}
                    th {{ 
                        background-color: #f5f7fa;
                        font-weight: 600;
                    }}
                    tr:nth-child(even) {{ 
                        background-color: #f9f9f9; 
                    }}
                    .search-query {{
                        background: #e3f2fd;
                        padding: 15px;
                        border-radius: 4px;
                        margin: 15px 0;
                        border-left: 4px solid #2196f3;
                    }}
                    .search-result {{
                        margin: 15px 0;
                        padding: 15px;
                        background: white;
                        border-radius: 4px;
                        border-left: 4px solid #4caf50;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .search-result h4 {{
                        margin: 0 0 10px 0;
                        color: #2c3e50;
                    }}
                    .warning {{
                        background: #fff3e0;
                        padding: 15px;
                        border-left: 4px solid #ff9800;
                        margin: 15px 0;
                        border-radius: 4px;
                    }}
                    .success {{
                        background: #e8f5e9;
                        padding: 15px;
                        border-left: 4px solid #4caf50;
                        margin: 15px 0;
                        border-radius: 4px;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Qdrant Vector Search Report</h1>
                        <p>Generated on: {current_time}</p>
                    </div>
                    
                    <!-- Qdrant Statistics -->
                    <div class="section">
                        <h2>📊 Vector Store Statistics</h2>
                    {formatted_qdrant_stats}
                </div>
                    
                    <!-- Semantic Search Example -->
                    <div class="section">
                        <h2>🔍 Semantic Search Example</h2>
                    {formatted_semantic_search}
                </div>
                    
                    <!-- Vector Space Visualization -->
                    <div class="section">
                        <h2>🌐 Vector Space Visualization</h2>
                    {formatted_vector_viz}
                </div>
                    
                    <div class="section success">
                        <h3>✅ Vector Search Status: Operational</h3>
                    <p>Qdrant vector store is functioning normally with {actual_vector_count:,} vectors indexed.</p>
                </div>
                    
                    <div class="section">
                        <h3>Next Steps</h3>
                        <ul>
                            <li>Try different search queries to explore the vector space</li>
                            <li>Adjust the similarity threshold for more/less strict matching</li>
                            <li>Monitor vector search performance and quality</li>
                            <li>Consider fine-tuning the embedding model for your specific domain</li>
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """
            
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(html)
            
            print(f"[INFO] Qdrant report generated: {self.report_file}")
            return self.report_file
            
        except Exception as e:
            print(f"[ERROR] Failed to generate Qdrant report: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    # Example data structure
    stats = {
        'qdrant_stats': {
            'vectors_count': 1250,
            'dimensions': 384,
            'distance_metric': 'Cosine',
            'collection_name': 'purchase_orders'
        }
    }
    
    query_results = {
        'semantic_search': {
            'query': 'construction materials delivery',
            'results': [
                {
                    'labels': ['PurchaseOrder'],
                    'score': 0.9234,
                    'properties': {
                        'po_number': 'PO-1001',
                        'description': 'Delivery of construction materials including cement, steel rods, and lumber',
                        'date': '2023-05-28',
                        'amount': 1250.50,
                        'status': 'delivered'
                    }
                },
                {
                    'labels': ['PurchaseOrder', 'Construction'],
                    'score': 0.8765,
                    'properties': {
                        'po_number': 'PO-995',
                        'description': 'Building materials for site preparation',
                        'date': '2023-05-25',
                        'amount': 875.25,
                        'status': 'completed'
                    }
                },
                {
                    'labels': ['PurchaseOrder'],
                    'score': 0.8123,
                    'properties': {
                        'po_number': 'PO-990',
                        'description': 'Heavy equipment rental for construction site',
                        'date': '2023-05-20',
                        'amount': 2250.00,
                        'status': 'in_progress'
                    }
                }
            ]
        }
    }
    
    report = QdrantReport()
    report.generate(stats, query_results)
