import os
import sys
from flask import Flask, render_template, request, jsonify

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the GraphRAG class
from src.graphrag import GraphRAG

app = Flask(__name__)

# Initialize GraphRAG instance
try:
    graph_rag = GraphRAG()
    print("GraphRAG initialized successfully")
except Exception as e:
    print(f"Error initializing GraphRAG: {str(e)}")
    graph_rag = None

@app.route('/')
def index():
    """Render the main search page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests from the frontend."""
    if not graph_rag:
        return jsonify({
            'success': False, 
            'error': 'GraphRAG system is not properly initialized. Please check the server logs.'
        })
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        limit = int(data.get('limit', 5))
        project_name = data.get('project_name')
        supplier_name = data.get('supplier_name')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Search query cannot be empty.'
            })
        
        # Perform the search
        results = graph_rag.search_similar_pos(
            query=query,
            limit=limit,
            project_name=project_name,
            supplier_name=supplier_name
        )
        
        # Convert any non-serializable objects to strings
        processed_results = []
        for result in results:
            processed_result = {}
            for key, value in result.items():
                # Convert any non-serializable objects to strings
                processed_result[key] = str(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
            processed_results.append(processed_result)
        
        return jsonify({
            'success': True,
            'results': processed_results
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in search: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_trace
        }), 500

if __name__ == '__main__':
    # Check if the required environment variables are set
    required_vars = [
        'NEO4J_URI',
        'NEO4J_USER',
        'NEO4J_PASSWORD',
        'QDRANT_URL',
        'QDRANT_COLLECTION_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Warning: The following required environment variables are not set: {', '.join(missing_vars)}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
