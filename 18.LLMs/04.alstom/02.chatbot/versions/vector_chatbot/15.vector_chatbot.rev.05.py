"""
Alstom Project Assistant - Vector-based Chatbot Interface (Network Version)
This script provides a web interface for the Alstom Project Assistant.
It uses pre-processed document embeddings to retrieve relevant documents.
This version uses network paths to access documents stored on the file server.
"""

import os
import json
import time
import numpy as np
import requests
import urllib.parse
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VectorChatbot")

# Define paths
VECTOR_DB_DIR = "C:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\05.llm\\alstom"
DOCUMENTS_DIR = os.path.join(VECTOR_DB_DIR, "documents")
EMBEDDINGS_FILE = os.path.join(VECTOR_DB_DIR, "document_embeddings.json")
DOCUMENTS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "document_index.json")

# Ollama API configuration
OLLAMA_API_URL = "http://10.10.12.202:11434"  # IP address of the GPU machine
MODEL_NAME = "mistral"  # Using mistral model which is available on the server

# Global variables
OLLAMA_AVAILABLE = False
OLLAMA_WORKING = False
EMBEDDING_MODEL = None
LAST_CONNECTION_CHECK = 0
CONNECTION_CHECK_INTERVAL = 30  # Check connection every 30 seconds
document_index = {}
document_embeddings = {}

# Check if Ollama API is available
def check_ollama_availability():
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if model.get("name") == MODEL_NAME:
                    return True
            logger.warning(f"Model {MODEL_NAME} not found in Ollama")
            return False
        else:
            logger.warning(f"Ollama API returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Ollama API: {str(e)}")
        return False

# Test Ollama connection with a simple prompt
def test_ollama_connection():
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "Hello, are you working?",
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return True, response.json().get("response", "")
        else:
            logger.warning(f"Ollama generate API returned status code {response.status_code}")
            return False, ""
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {str(e)}")
        return False, ""

# Initialize the embedding model
def initialize_embedding_model():
    try:
        # Try to load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded embedding model")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        
        # Try to install the model
        try:
            logger.info("Attempting to install sentence-transformers...")
            import subprocess
            subprocess.check_call(["pip", "install", "sentence-transformers"])
            
            # Try loading again
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully installed and loaded embedding model")
            return model
        except Exception as e2:
            logger.error(f"Failed to install sentence-transformers: {str(e2)}")
            return None

# Load document index
def load_document_index():
    if os.path.exists(DOCUMENTS_INDEX_FILE):
        try:
            with open(DOCUMENTS_INDEX_FILE, 'r', encoding='utf-8') as f:
                index = json.load(f)
            logger.info(f"Loaded document index with {len(index)} documents")
            return index
        except Exception as e:
            logger.error(f"Error loading document index: {str(e)}")
            return {}
    else:
        logger.warning(f"Document index file not found: {DOCUMENTS_INDEX_FILE}")
        return {}

# Load document embeddings
def load_document_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                embeddings = json.load(f)
            
            # Convert string keys to tuples if needed
            if all(isinstance(k, str) for k in embeddings.keys()):
                embeddings = {eval(k): np.array(v) for k, v in embeddings.items()}
            
            logger.info(f"Loaded embeddings for {len(embeddings)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading document embeddings: {str(e)}")
            return {}
    else:
        logger.warning(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        return {}

# Get embedding for a text
def get_embedding(text):
    if EMBEDDING_MODEL is None:
        logger.error("Embedding model not initialized")
        return None
    
    try:
        return EMBEDDING_MODEL.encode(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

# Find relevant documents based on query
def find_relevant_documents(query, top_k=3):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []
    
    # Calculate similarity with all documents
    similarities = {}
    for doc_id, doc_embedding in document_embeddings.items():
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        similarities[doc_id] = similarity
    
    # Sort by similarity
    sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Get top k documents
    relevant_docs = []
    for i, (doc_id, similarity) in enumerate(sorted_docs[:top_k]):
        if doc_id in document_index:
            doc_info = document_index[doc_id].copy()
            doc_info['similarity'] = float(similarity)
            relevant_docs.append(doc_info)
    
    return relevant_docs

# Generate response from Ollama
def generate_response(prompt, system_prompt=None):
    try:
        # Prepare the request
        request_data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Send the request
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=request_data
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            logger.error(f"Ollama API returned status code {response.status_code}")
            return f"Error: Ollama API returned status code {response.status_code}"
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

# Function to check connection status
def check_connection_status():
    global OLLAMA_AVAILABLE, OLLAMA_WORKING
    try:
        OLLAMA_AVAILABLE = check_ollama_availability()
        if OLLAMA_AVAILABLE:
            OLLAMA_WORKING, _ = test_ollama_connection()
        else:
            OLLAMA_WORKING = False
        
        logger.info(f"Connection status updated: Available={OLLAMA_AVAILABLE}, Working={OLLAMA_WORKING}")
        
        # Return status as JSON for the JavaScript to use
        return {"connected": OLLAMA_WORKING}
    except Exception as e:
        logger.error(f"Error checking connection status: {str(e)}")
        return {"connected": False}

# Process user question
def process_question(message, history, doc_paths_state):
    try:
        # Check Ollama connection status periodically
        global OLLAMA_AVAILABLE, OLLAMA_WORKING, LAST_CONNECTION_CHECK
        current_time = time.time()
        
        if current_time - LAST_CONNECTION_CHECK > CONNECTION_CHECK_INTERVAL:
            LAST_CONNECTION_CHECK = current_time
            OLLAMA_AVAILABLE = check_ollama_availability()
            if OLLAMA_AVAILABLE:
                OLLAMA_WORKING, _ = test_ollama_connection()
            else:
                OLLAMA_WORKING = False
            logger.info(f"Connection status updated: Available={OLLAMA_AVAILABLE}, Working={OLLAMA_WORKING}")
        
        if not message or message.strip() == "":
            return history, doc_paths_state, ""
            
        # Find relevant documents
        relevant_docs = find_relevant_documents(message)
        
        # Update document paths state
        doc_paths = doc_paths_state.copy()
        doc_paths.extend([doc['path'] for doc in relevant_docs])
        
        # Create context from relevant documents
        context = ""
        for i, doc in enumerate(relevant_docs):
            context += f"Document {i+1}: {doc['name']} (in {doc['simplified_path']})\n"
            context += f"Content: {doc.get('content', 'No content available')}\n\n"
        
        # Create prompt with context
        prompt = f"""
        The user is asking about the Alstom project. Here are some relevant documents:
        
        {context}
        
        User question: {message}
        
        Please provide a helpful response based on the information in these documents. If the documents don't contain relevant information, say so and provide general guidance.
        """
        
        # Check if Ollama is available
        if not OLLAMA_AVAILABLE or not OLLAMA_WORKING:
            logger.error("Ollama API is not available or not working")
            new_history = history.copy()
            new_history.append([message, "Error: Cannot connect to the GPU machine running Ollama. Please check the connection and try again."])
            return new_history, doc_paths, ""
        
        # Generate response
        try:
            response_text = generate_response(prompt)
        except Exception as api_error:
            logger.error(f"Error calling Ollama API: {str(api_error)}")
            new_history = history.copy()
            new_history.append([message, f"Error connecting to Ollama API: {str(api_error)}"])
            return new_history, doc_paths, ""
        
        # Format the response with document sources
        formatted_response = response_text + "\n\nSources:\n"
        for i, doc in enumerate(relevant_docs):
            formatted_response += f"- {doc['name']} (in {doc['simplified_path']})\n"
        
        # Create HTML for document buttons with multiple access options
        doc_buttons_html = "<div style='margin-top: 10px;'><strong>Document References:</strong><br>"
        
        # Document server configuration
        DOC_SERVER_PORT = 7861
        DOC_SERVER_HOST = "10.10.11.244"  # Same as the chatbot server
        
        for i, doc in enumerate(relevant_docs):
            doc_path = doc['local_path'] if doc['local_path'] else doc['path']
            doc_name = doc['name']
            simplified_path = doc['simplified_path']
            
            # Convert local path to network path
            network_path = doc_path.replace("C:\\alstom", "\\\\fileserver2\\Head Office Server\\Projects Control (PC)\\10 Backup\\05 Models\\alstom")
            
            # Create URL for document server
            doc_server_url = f"http://{DOC_SERVER_HOST}:{DOC_SERVER_PORT}/document?path={urllib.parse.quote(network_path)}"
            
            # Create a document card with multiple access options
            doc_buttons_html += f'''
            <div style='margin: 5px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background-color: #f9f9f9;'>
                <div style='font-weight: bold;'>{doc_name}</div>
                <div style='font-size: 0.8em; color: #666; margin-bottom: 5px;'>Location: {simplified_path}</div>
                <div>
                    <a href="{doc_server_url}" download="{doc_name}" target="_blank"
                       style='display: inline-block; margin-right: 5px; padding: 3px 8px; background-color: #4CAF50; border: 1px solid #45a049; color: white; border-radius: 3px; cursor: pointer; text-decoration: none;'>
                        Download
                    </a>
                    <button onclick="window.open('{doc_server_url}', '_blank')" 
                            style='margin-right: 5px; padding: 3px 8px; background-color: #007bff; border: 1px solid #0069d9; color: white; border-radius: 3px; cursor: pointer;'>
                        View Online
                    </button>
                    <button onclick="window.open('file:///{network_path.replace('\\', '/')}', '_blank')" 
                            style='margin-right: 5px; padding: 3px 8px; background-color: #e0e0e0; border: 1px solid #ccc; border-radius: 3px; cursor: pointer;'>
                        Open File
                    </button>
                    <button onclick="navigator.clipboard.writeText('{network_path}'); alert('Path copied to clipboard!')" 
                            style='padding: 3px 8px; background-color: #e0e0e0; border: 1px solid #ccc; border-radius: 3px; cursor: pointer;'>
                        Copy Path
                    </button>
                </div>
            </div>
            '''
        
        # Add instructions for accessing documents
        doc_buttons_html += '''
        <div style='margin-top: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background-color: #f5f5f5;'>
            <div style='font-weight: bold; margin-bottom: 5px;'>How to Access Documents:</div>
            <ol style='margin: 0; padding-left: 20px; font-size: 0.9em;'>
                <li>Use the <strong>Download</strong> button (green) to save the document to your computer</li>
                <li>Use the <strong>View Online</strong> button (blue) to view documents directly in your browser</li>
                <li>If you prefer to open the file directly from the network, use <strong>Open File</strong></li>
                <li>If other options don't work, use <strong>Copy Path</strong> and paste in File Explorer</li>
                <li>Make sure the document server is running on port 7861</li>
            </ol>
        </div>
        </div>'''
        
        # Add the new message pair to history
        new_history = history.copy()
        new_history.append([message, formatted_response])
        return new_history, doc_paths, doc_buttons_html
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        new_history = history.copy()
        new_history.append([message, f"Error processing your question: {str(e)}"])
        return new_history, doc_paths_state, ""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(css="""
        footer {visibility: hidden}
        .left-panel { border-right: 1px solid #e0e0e0; }
        .status-connected { color: green; font-weight: bold; }
        .status-disconnected { color: red; font-weight: bold; }
        .input-area { position: fixed; bottom: 0; left: 25%; right: 0; background: white; padding: 10px; z-index: 1000; }
        .chat-container { margin-bottom: 60px; }
        .logo-container { text-align: center; margin-bottom: 15px; }
        .logo-container img { max-width: 150px; margin-bottom: 10px; }
    """) as demo:
        # Store document paths for the current conversation
        doc_paths = gr.State([])
        
        with gr.Row():
            # Left panel (1/4 of screen)
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                # Company logo at the top followed by Alstom logo
                with gr.Column(elem_classes=["logo-container"]):
                    gr.Image(os.path.join(os.path.dirname(__file__), "static", "rme.png"), 
                             show_label=False, 
                             height=80)
                    gr.Image(os.path.join(os.path.dirname(__file__), "static", "logo.svg"), 
                             show_label=False, 
                             height=60)
                
                # GPU Status indicator
                if OLLAMA_AVAILABLE and OLLAMA_WORKING:
                    status_class = "status-connected"
                    status_text = "✓ Connected to NVIDIA 4090 GPU"
                else:
                    status_class = "status-disconnected"
                    status_text = "✗ Cannot connect to GPU machine"
                
                # Create connection status element with a unique ID for JavaScript to update
                connection_status = gr.HTML(
                    f"<div id='gpu-status' class='{status_class}' style='margin-top: 10px; padding: 10px;'>{status_text}</div>"
                )
                
                # Document stats will be updated dynamically
                gr.Markdown(f"Documents: {len(document_index)}")
                
                # Placeholder for future links
                gr.Markdown("""
                ### Navigation
                *Future links will appear here*
                """)
            
            # Right panel (3/4 of screen - chat interface)
            with gr.Column(scale=3):
                # Title
                gr.Markdown(
                    """
                    # Alstom Project Assistant
                    Ask questions about the Alstom project documents and get answers based on the project documentation.
                    """
                )
                
                # Chat interface
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=500,
                    elem_classes=["chat-container"]
                )
                
                # Document buttons container
                doc_buttons = gr.HTML()
                
                # Input area
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about the Alstom project...",
                        show_label=False,
                        container=False
                    )
                    submit = gr.Button("Send")
                
                # Set up event handlers
                submit_event = msg.submit(
                    process_question, 
                    [msg, chatbot, doc_paths], 
                    [chatbot, doc_paths, doc_buttons],
                    api_name="submit"
                )
                submit_click_event = submit.click(
                    process_question, 
                    [msg, chatbot, doc_paths], 
                    [chatbot, doc_paths, doc_buttons],
                    api_name="submit_click"
                )
                
                # Clear the input box after submission
                submit_event.then(lambda: "", None, msg)
                submit_click_event.then(lambda: "", None, msg)
        
        return demo

def main():
    global OLLAMA_AVAILABLE, OLLAMA_WORKING, EMBEDDING_MODEL, document_index, document_embeddings
    
    # Check Ollama availability
    logger.info(f"Checking connection to Ollama API at {OLLAMA_API_URL}...")
    OLLAMA_AVAILABLE = check_ollama_availability()
    
    if OLLAMA_AVAILABLE:
        logger.info("✓ Successfully connected to Ollama API and found required model")
        
        # Test Ollama connection
        logger.info("Testing Ollama with a simple prompt...")
        OLLAMA_WORKING, test_response = test_ollama_connection()
        
        if OLLAMA_WORKING:
            logger.info("✓ Successfully tested Ollama generate API")
        else:
            logger.error("✗ Failed to test Ollama generate API")
    else:
        logger.error("✗ Failed to connect to Ollama API or model not found")
    
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    EMBEDDING_MODEL = initialize_embedding_model()
    
    # Load document index
    logger.info("Loading document index...")
    document_index = load_document_index()
    
    # Load document embeddings
    logger.info("Loading document embeddings...")
    document_embeddings = load_document_embeddings()
    
    # Create the interface
    demo = create_interface()
    
    # Add JavaScript to periodically update connection status
    js_code = """
    function updateConnectionStatus() {
        fetch('/connection-status')
            .then(response => response.json())
            .then(data => {
                const statusElement = document.getElementById('gpu-status');
                if (statusElement) {
                    if (data.connected) {
                        statusElement.className = 'status-connected';
                        statusElement.innerHTML = '✓ Connected to NVIDIA 4090 GPU';
                    } else {
                        statusElement.className = 'status-disconnected';
                        statusElement.innerHTML = '✗ Cannot connect to GPU machine';
                    }
                }
            })
            .catch(error => console.error('Error checking connection:', error));
    }
    
    // Update status every 30 seconds
    setInterval(updateConnectionStatus, 30000);
    """
    
    # Launch the interface
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
