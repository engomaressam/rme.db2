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
                # Assuming 'payload' contains the properties we want to display
                payload = result.get('payload', {})
                doc_id = result.get('id', 'N/A')
                score = f"{result.get('score', 0):.4f}" if 'score' in result else 'N/A'
                
                # Extract relevant fields from payload, e.g., 'text' or 'filename'
                # This part is highly dependent on your Qdrant point structure
                display_text = payload.get('text_chunk', payload.get('text', 'No text available'))
                if len(display_text) > 200: # Truncate long text
                    display_text = display_text[:200] + '...'

                source_info = ""
                if 'filename' in payload:
                    source_info += f"<div><strong>Source:</strong> {payload['filename']}</div>"
                if 'page_number' in payload:
                     source_info += f"<div><strong>Page:</strong> {payload['page_number']}</div>"
                if 'chunk_index' in payload:
                    source_info += f"<div><strong>Chunk:</strong> {payload['chunk_index']}</div>"

                html += f"""
                <div class="search-result">
                    <h4>Result {i}: ID {doc_id} <span style="color: #666;">(Score: {score})</span></h4>
                    <div style="margin-left: 15px; margin-top: 5px;">
                        <p><strong>Content Snippet:</strong> {display_text}</p>
                        {source_info}
                    </div>
                </div>"""
            except Exception as e:
                print(f"[WARNING] Error formatting search result: {e}")
        
        return html
    
    def _format_semantic_search(self, query_results: Dict[str, Any]) -> str:
        """Format semantic search section"""
        semantic_data = query_results.get('semantic_search_results', {}) # Adjusted key
        if not semantic_data: # Check if semantic_data itself is empty or None
            return '<div class="warning">No semantic search results available</div>'
        
        query_text = semantic_data.get('query', 'N/A')
        results = semantic_data.get('results', [])

        return f"""
        <div class="search-query">
            <h3>Semantic Search Query</h3>
            <p>{query_text}</p>
        </div>
        <div class="search-results">
            <h4>Top Results:</h4>
            {self._format_search_results(results)}
        </div>
        """

    def _format_nearest_neighbors(self, query_results: Dict[str, Any]) -> str:
        """Format nearest neighbors section"""
        nn_data = query_results.get('nearest_neighbors_results', {}) # Adjusted key
        if not nn_data:
            return '<div class="warning">No nearest neighbor results available</div>'

        source_id = nn_data.get('source_id', 'N/A')
        source_vector_retrieved = nn_data.get('source_vector_retrieved', False)
        results = nn_data.get('results', [])

        source_status = "Retrieved" if source_vector_retrieved else "Not found or no vector"

        return f"""
        <div class="search-query">
            <h3>Nearest Neighbors for ID: {source_id}</h3>
            <p><strong>Source Vector Status:</strong> {source_status}</p>
        </div>
        <div class="search-results">
            <h4>Top Neighbors:</h4>
            {self._format_search_results(results)}
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
            # Use the correct keys from system_demo.py for query_results
            formatted_semantic_search = self._format_semantic_search(query_results) 
            formatted_nearest_neighbors = self._format_nearest_neighbors(query_results)
            formatted_vector_viz = self._format_vector_visualization()
            actual_vector_count = stats.get('qdrant_stats', {}).get('vectors_count', 0)
            
            html_content = f"""
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
                    .search-query {{                        background: #e3f2fd;
                        padding: 15px;
                        border-radius: 4px;
                        margin: 15px 0;
                        border-left: 4px solid #2196f3;
                    }}
                    .search-result {{                        margin: 15px 0;
                        padding: 15px;
                        background: white;
                        border-radius: 4px;
                        border-left: 4px solid #4caf50;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .search-result h4 {{                        margin: 0 0 10px 0;
                        color: #2c3e50;
                    }}
                    .warning {{                        background: #fff3e0;
                        padding: 15px;
                        border-left: 4px solid #ff9800;
                        margin: 15px 0;
                        border-radius: 4px;
                    }}
                    .success {{                        background: #e8f5e9;
                        padding: 15px;
                        border-left: 4px solid #28a745;
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
                    
                    <!-- Qdrant Statistics Section -->
                    <div class="section">
                        <h2>Qdrant Collection Statistics</h2>
                        <p>Collection Name: <strong>{stats.get('qdrant_stats', {}).get('collection_name', 'N/A')}</strong></p>
                        <p>Total Vectors: <strong>{actual_vector_count:,}</strong></p>
                        {formatted_qdrant_stats}
                    </div>

                    <!-- Semantic Search Section -->
                    <div class="section">
                        <h2>Semantic Search</h2>
                        {formatted_semantic_search}
                    </div>

                    <!-- Nearest Neighbors Section -->
                    <div class="section">
                        <h2>Nearest Neighbors Search</h2>
                        {formatted_nearest_neighbors}
                    </div>
                    
                    <!-- Vector Visualization Placeholder -->
                    <div class="section">
                        {formatted_vector_viz}
                    </div>
                    
                    <div class="section success">
                        <p>Report generated successfully.</p>
                    </div>
                </div>
            </body>
            </html>
            """

            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"[INFO] Qdrant report generated: {self.report_file}")
            return self.report_file
        except Exception as e:
            print(f"[ERROR] Failed to generate Qdrant report: {e}")
            error_html = f"<html><body><h1>Error</h1><p>Failed to generate Qdrant report: {e}</p></body></html>"
            try:
                with open(self.report_file, 'w', encoding='utf-8') as f:
                    f.write(error_html)
                print(f"[INFO] Qdrant error report generated: {self.report_file}")
            except Exception as ef:
                print(f"[ERROR] Failed to write Qdrant error report: {ef}")
            return self.report_file

# Example Usage (Optional - for testing the report class directly)
# if __name__ == "__main__":
#     sample_stats_qdrant = {{
#         'qdrant_stats': {{
#             'collection_name': 'my_collection',
#             'vectors_count': 1000,
#             'dimensions': 384,
#             'distance_metric': 'Cosine'
#         }}
#     }}
#     sample_query_results_qdrant = {{
#         'semantic_search_results': {{
#             'query': 'search for similar items',
#             'results': [
#                 {{'id': 'doc1', 'score': 0.95, 'payload': {{'text': 'This is document 1 about apples.'}}}},
#                 {{'id': 'doc2', 'score': 0.92, 'payload': {{'text': 'Another document, this one about oranges.'}}}}
#             ]
#         }},
#         'nearest_neighbors_results': {{
#             'source_id': 'doc_xyz',
#             'source_vector_retrieved': True,
#             'results': [
#                 {{'id': 'doc3', 'score': 0.88, 'payload': {{'text': 'Document 3, similar to doc_xyz.'}}}},
#                 {{'id': 'doc4', 'score': 0.85, 'payload': {{'text': 'Document 4, also quite similar.'}}}}
#             ]
#         }}
#     }}
#     q_report_generator = QdrantReport(report_dir='system_reports_test_qdrant')
#     q_report_generator.generate(sample_stats_qdrant, sample_query_results_qdrant)
