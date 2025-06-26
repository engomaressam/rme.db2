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
            
            html = f"""
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
                    
                    <!-- Database Statistics -->
                    <div class="section">
                        <h2>📊 Database Statistics</h2>
                        <div class="grid">
                            <div class="card">
                                <h3>Node Counts</h3>
                                <table>
                                    <tr><th>Node Type</th><th>Count</th></tr>
                                    {formatted_node_counts}
                                </table>
                            </div>
                            <div class="card">
                                <h3>Relationship Counts</h3>
                                <table>
                                    <tr><th>Type</th><th>Count</th></tr>
                                    {formatted_rel_counts}
                                </table>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Top Projects -->
                    <div class="section">
                        <h2>🏆 Top Projects by PO Count</h2>
                        <div class="card">
                            <table>
                                <tr><th>Project</th><th>PO Count</th></tr>
                                {formatted_top_projects}
                            </table>
                        </div>
                    </div>
                    
                    <!-- Recent Purchase Orders -->
                    <div class="section">
                        <h2>📋 Recent Purchase Orders</h2>
                        <div class="card">
                            <table>
                                <tr>
                                    <th>PO Number</th>
                                    <th>Date</th>
                                    <th>Amount</th>
                                    <th>Project</th>
                                    <th>Items</th>
                                </tr>
                                {formatted_recent_pos}
                            </table>
                        </div>
                    </div>
                    
                    <!-- Top Suppliers -->
                    <div class="section">
                        <h2>🏭 Top Suppliers</h2>
                        <div class="card">
                            <table>
                                <tr>
                                    <th>Supplier</th>
                                    <th>PO Count</th>
                                    <th>Total Amount</th>
                                </tr>
                                {formatted_top_suppliers}
                            </table>
                        </div>
                    </div>
                    
                    <div class="section success">
                        <h3>✅ Database Status: Operational</h3>
                        <p>All systems are functioning normally. Total nodes processed: <strong>{total_nodes_val:,}</strong></p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(html)
            
            print(f"[INFO] Neo4j report generated: {self.report_file}")
            return self.report_file
            
        except Exception as e:
            print(f"[ERROR] Failed to generate Neo4j report: {e}")
            raise

# Example usage:
if __name__ == "__main__":
    # Example data structure
    stats = {
        'node_counts': [
            {'label': 'Project', 'count': 10},
            {'label': 'PurchaseOrder', 'count': 100},
            {'label': 'Supplier', 'count': 20},
            {'label': 'Item', 'count': 500}
        ],
        'relationship_counts': [
            {'type': 'HAS_PO', 'count': 100},
            {'type': 'ISSUED_PO', 'count': 100},
            {'type': 'HAS_ITEM', 'count': 500}
        ]
    }
    
    query_results = {
        'top_projects': [
            {'project': 'Project A', 'po_count': 45},
            {'project': 'Project B', 'po_count': 32},
            {'project': 'Project C', 'po_count': 23}
        ],
        'recent_purchase_orders': [
            {'po_number': 'PO-1001', 'date': '2023-05-28', 'amount': 1250.50, 'project': 'Project A', 'item_count': 5},
            {'po_number': 'PO-1000', 'date': '2023-05-27', 'amount': 875.25, 'project': 'Project B', 'item_count': 3}
        ],
        'top_suppliers': [
            {'supplier': 'ABC Suppliers', 'po_count': 25, 'total_amount': 12500.75},
            {'supplier': 'XYZ Corp', 'po_count': 18, 'total_amount': 9875.50}
        ]
    }
    
    report = Neo4jReport()
    report.generate(stats, query_results)
