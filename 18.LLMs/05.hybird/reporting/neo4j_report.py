import os
from datetime import datetime
from typing import Dict, Any

class Neo4jReport:
    def __init__(self, report_dir: str):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.report_file = os.path.join(self.report_dir, f"neo4j_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    
    def _format_node_counts(self, stats: Dict[str, Any]) -> str:
        """Format node counts for HTML report"""
        if not stats.get('node_counts'):
            return '<tr><td colspan="2">No node data available</td></tr>'
        
        html = ""
        for node in stats['node_counts']:
            try:
                label = node.get('label', 'Unknown')
                if isinstance(label, list):
                    label = ', '.join(label) if label else 'No Label'
                count = node.get('count', 0)
                html += f'<tr><td>{label}</td><td style="text-align: right;">{count:,}</td></tr>'
            except Exception as e:
                print(f"[WARNING] Error formatting node count: {e}")
        return html
    
    def _format_relationship_counts(self, stats: Dict[str, Any]) -> str:
        """Format relationship counts for HTML report"""
        if not stats.get('relationship_counts'):
            return '<tr><td colspan="2">No relationship data available</td></tr>'
        
        html = ""
        for rel in stats['relationship_counts']:
            try:
                rel_type = rel.get('type', 'Unknown')
                count = rel.get('count', 0)
                html += f'<tr><td>{rel_type}</td><td style="text-align: right;">{count:,}</td></tr>'
            except Exception as e:
                print(f"[WARNING] Error formatting relationship: {e}")
        return html
    
    def _format_top_projects(self, query_results: Dict[str, Any]) -> str:
        """Format top projects for HTML report"""
        if not query_results.get('top_projects'):
            return '<tr><td colspan="2">No project data available</td></tr>'
        
        html = ""
        for proj in query_results['top_projects']:
            try:
                name = proj.get('project', 'Unknown')
                count = proj.get('po_count', 0)
                html += f'<tr><td>{name}</td><td style="text-align: right;">{count:,}</td></tr>'
            except Exception as e:
                print(f"[WARNING] Error formatting project: {e}")
        return html
    
    def _format_recent_purchase_orders(self, query_results: Dict[str, Any]) -> str:
        """Format recent purchase orders for HTML report"""
        if not query_results.get('recent_purchase_orders'):
            return '<tr><td colspan="5">No purchase orders available</td></tr>'
        
        html = ""
        for po in query_results['recent_purchase_orders']:
            try:
                po_num = po.get('po_number', 'N/A')
                date = po.get('date', 'N/A')
                amount = f"${float(po.get('amount', 0)):,.2f}" if po.get('amount') else 'N/A'
                project = po.get('project', 'N/A')
                items = po.get('item_count', 0)
                
                html += f'''
                <tr>
                    <td>{po_num}</td>
                    <td>{date}</td>
                    <td style="text-align: right;">{amount}</td>
                    <td>{project}</td>
                    <td style="text-align: right;">{items}</td>
                </tr>'''
            except Exception as e:
                print(f"[WARNING] Error formatting PO: {e}")
        return html
    
    def _format_top_suppliers(self, query_results: Dict[str, Any]) -> str:
        """Format top suppliers for HTML report"""
        if not query_results.get('top_suppliers'):
            return '<tr><td colspan="3">No supplier data available</td></tr>'
        
        html = ""
        for supplier in query_results['top_suppliers']:
            try:
                name = supplier.get('supplier', 'Unknown')
                count = supplier.get('po_count', 0)
                amount = f"${float(supplier.get('total_amount', 0)):,.2f}" if supplier.get('total_amount') else 'N/A'
                
                html += f'''
                <tr>
                    <td>{name}</td>
                    <td style="text-align: right;">{count:,}</td>
                    <td style="text-align: right;">{amount}</td>
                </tr>'''
            except Exception as e:
                print(f"[WARNING] Error formatting supplier: {e}")
        return html
    
    def generate(self, stats: Dict[str, Any], query_results: Dict[str, Any]) -> str:
        """Generate the Neo4j HTML report"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Pre-format all dynamic HTML parts
            formatted_node_counts = self._format_node_counts(stats)
            formatted_rel_counts = self._format_relationship_counts(stats)
            formatted_top_projects = self._format_top_projects(query_results)
            formatted_recent_pos = self._format_recent_purchase_orders(query_results)
            formatted_top_suppliers = self._format_top_suppliers(query_results)
            total_nodes_val = sum(node.get('count', 0) for node in stats.get('node_counts', []))
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Neo4j Database Report</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        line-height: 1.6; 
                        margin: 0;
                        padding: 20px;
                        color: #333;
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
                        padding: 15px; 
                        margin-bottom: 15px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    h1, h2, h3, h4 {{ 
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
                    .success {{ 
                        color: #28a745; 
                        padding: 10px;
                        background: #e8f5e9;
                        border-left: 4px solid #28a745;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Neo4j Database Report</h1>
                        <p>Generated on: {current_time}</p>
                    </div>
                    
                    <!-- Database Statistics Section -->
                    <div class="section">
                        <h2>Database Statistics</h2>
                        <div class="grid">
                            <div class="card">
                                <h4>Total Nodes</h4>
                                <p style="font-size: 24px; font-weight: bold;">{total_nodes_val:,}</p>
                            </div>
                            <div class="card">
                                <h4>Node Counts by Label</h4>
                                <table>
                                    <thead><tr><th>Label</th><th>Count</th></tr></thead>
                                    <tbody>{formatted_node_counts}</tbody>
                                </table>
                            </div>
                            <div class="card">
                                <h4>Relationship Counts by Type</h4>
                                <table>
                                    <thead><tr><th>Type</th><th>Count</th></tr></thead>
                                    <tbody>{formatted_rel_counts}</tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    <!-- Query Results Section -->
                    <div class="section">
                        <h2>Query Results</h2>

                        <div class="card">
                            <h3>Top Projects by Purchase Order Count</h3>
                            <table>
                                <thead><tr><th>Project Name</th><th>PO Count</th></tr></thead>
                                <tbody>{formatted_top_projects}</tbody>
                            </table>
                        </div>

                        <div class="card">
                            <h3>Recent Purchase Orders</h3>
                            <table>
                                <thead><tr><th>PO Number</th><th>Date</th><th>Amount</th><th>Project</th><th>Item Count</th></tr></thead>
                                <tbody>{formatted_recent_pos}</tbody>
                            </table>
                        </div>
                        
                        <div class="card">
                            <h3>Top Suppliers by Purchase Order Value</h3>
                             <table>
                                <thead><tr><th>Supplier Name</th><th>PO Count</th><th>Total Amount</th></tr></thead>
                                <tbody>{formatted_top_suppliers}</tbody>
                            </table>
                        </div>
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
            print(f"[INFO] Neo4j report generated: {self.report_file}")
            return self.report_file
        except Exception as e:
            print(f"[ERROR] Failed to generate Neo4j report: {e}")
            error_html = f"<html><body><h1>Error</h1><p>Failed to generate Neo4j report: {e}</p></body></html>"
            try:
                with open(self.report_file, 'w', encoding='utf-8') as f:
                    f.write(error_html)
                print(f"[INFO] Neo4j error report generated: {self.report_file}")
            except Exception as ef: 
                print(f"[ERROR] Failed to write Neo4j error report: {ef}")
            return self.report_file

# Example Usage (Optional - for testing the report class directly)
# if __name__ == "__main__":
#     sample_stats = {{
#         'node_counts': [
#             {{'label': 'Project', 'count': 10}},
#             {{'label': 'PurchaseOrder', 'count': 150}},
#             {{'label': 'Supplier', 'count': 25}},
#             {{'label': 'Item', 'count': 500}}
#         ],
#         'relationship_counts': [
#             {{'type': 'HAS_PO', 'count': 150}},
#             {{'type': 'SUPPLIED_BY', 'count': 150}},
#             {{'type': 'CONTAINS_ITEM', 'count': 700}}
#         ]
#     }}
#     sample_query_results = {{
#         'top_projects': [
#             {{'project': 'Project Alpha', 'po_count': 50}},
#             {{'project': 'Project Beta', 'po_count': 40}}
#         ],
#         'recent_purchase_orders': [
#             {{'po_number': 'PO-001', 'date': '2023-10-01', 'amount': '1000.00', 'project': 'Project Alpha', 'item_count': 5}},
#             {{'po_number': 'PO-002', 'date': '2023-10-05', 'amount': '2500.50', 'project': 'Project Beta', 'item_count': 10}}
#         ],
#         'top_suppliers': [
#             {{'supplier': 'Supplier X', 'po_count': 30, 'total_amount': '50000.00'}},
#             {{'supplier': 'Supplier Y', 'po_count': 20, 'total_amount': '45000.75'}}
#         ]
#     }}
#     report_generator = Neo4jReport(report_dir='system_reports_test_neo4j')
#     report_generator.generate(sample_stats, sample_query_results)
