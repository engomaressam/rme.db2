"""
Alstom Project Assistant - Vector-based Chatbot Interface (Network Version)
This script provides a web interface for the Alstom Project Assistant.
It uses pre-processed document embeddings to retrieve relevant documents.
This version includes an integrated document server for easy document access.
"""

import os
import json
import time
import threading
import numpy as np
import requests
import urllib.parse
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import mimetypes
import socket

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
OLLAMA_API_URL = "http://10.10.12.202:11434"  # Ollama API URL

# Try alternative URLs if the main one doesn't work
ALTERNATIVE_OLLAMA_URLS = [
    "http://10.10.12.202:11434",  # Original URL
    "http://localhost:11434",     # Local machine
    "http://127.0.0.1:11434"      # Localhost IP
]

# Available models
AVAILABLE_MODELS = {
    "Qwen 2.5 Coder (7B)": "qwen2.5-coder:7b",
    "Llama 4 (17B)": "llama4:17b-scout-16e-instruct-q4_K_M",
    "DeepSeek Coder (6.7B)": "deepseek-coder:6.7b",
    "Llama 3.2": "llama3.2:latest"
}

# Default model
DEFAULT_MODEL = "qwen2.5-coder:7b"  # Default model to use
MODEL_NAME = DEFAULT_MODEL  # Will be updated based on user selection

# Global variables
OLLAMA_AVAILABLE = False
OLLAMA_WORKING = False
EMBEDDING_MODEL = None
LAST_CONNECTION_CHECK = 0
CONNECTION_CHECK_INTERVAL = 30  # Check connection every 30 seconds
document_index = {}
document_embeddings = {}
DOC_SERVER_PORT = 7861
DOC_SERVER_HOST = "0.0.0.0"  # Listen on all interfaces
CHATBOT_PORT = None  # Will be determined dynamically
CURRENT_MODEL = DEFAULT_MODEL  # Track the currently selected model

# Document server handler
class DocumentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse the URL and query parameters
            parsed_path = urllib.parse.urlparse(self.path)
            
            # Handle document requests
            if parsed_path.path == '/document':
                query_params = urllib.parse.parse_qs(parsed_path.query)
                if 'path' in query_params:
                    file_path = query_params['path'][0]
                    
                    # Check if file exists
                    if not os.path.isfile(file_path):
                        self.send_error(404, f"File not found: {file_path}")
                        return
                    
                    # Get file size
                    file_size = os.path.getsize(file_path)
                    
                    # Determine content type
                    content_type, _ = mimetypes.guess_type(file_path)
                    if content_type is None:
                        content_type = 'application/octet-stream'
                    
                    # Set headers
                    self.send_response(200)
                    self.send_header('Content-Type', content_type)
                    self.send_header('Content-Length', str(file_size))
                    self.send_header('Content-Disposition', f'inline; filename="{os.path.basename(file_path)}"')
                    self.end_headers()
                    
                    # Send file in chunks to handle large files
                    with open(file_path, 'rb') as f:
                        chunk_size = 8192  # 8KB chunks
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                else:
                    self.send_error(400, "Missing 'path' parameter")
            # Handle health check
            elif parsed_path.path == '/health':
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Document server is running')
            else:
                self.send_error(404, "Not Found")
        except Exception as e:
            logger.error(f"Error in document server: {str(e)}")
            self.send_error(500, f"Internal Server Error: {str(e)}")
    
    def log_message(self, format, *args):
        # Override to use our logger
        logger.info(f"DocumentServer: {args[0]} {args[1]} {args[2]}")

# Start document server in a separate thread
def start_document_server():
    try:
        # Create a threaded HTTP server
        class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
            daemon_threads = True
        
        # Try to start the server
        server = ThreadedHTTPServer((DOC_SERVER_HOST, DOC_SERVER_PORT), DocumentHandler)
        logger.info(f"Starting document server on port {DOC_SERVER_PORT}")
        
        # Run the server in a separate thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        logger.info("Document server started successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to start document server: {str(e)}")
        return False

# Check if Ollama API is available
def check_ollama_availability(model_name=None):
    if model_name is None:
        model_name = CURRENT_MODEL
        
    try:
        # Set a timeout to avoid hanging
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            # Log all available models for debugging
            model_names = [model.get('name', '') for model in models]
            logger.info(f"Available models: {model_names}")
            
            # Check if any model contains our model name (more flexible matching)
            model_base_name = model_name.split(':')[0].lower()
            for model in models:
                model_full_name = model.get("name", "").lower()
                if model_base_name in model_full_name:
                    logger.info(f"Found matching model: {model.get('name')}")
                    return True
                    
            logger.warning(f"Model {model_name} not found in Ollama")
            return False
        else:
            logger.warning(f"Ollama API returned status code {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        logger.error(f"Timeout connecting to Ollama API at {OLLAMA_API_URL}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error connecting to Ollama API at {OLLAMA_API_URL}")
        return False
    except Exception as e:
        logger.error(f"Error connecting to Ollama API: {str(e)}")
        return False

# Test Ollama connection with a simple prompt
def test_ollama_connection(model_name=None):
    if model_name is None:
        model_name = CURRENT_MODEL
    
    # Set a longer timeout for larger models
    timeout = 60 if "mixtral" in model_name.lower() else 15
    logger.info(f"Testing connection to model {model_name} with timeout {timeout}s")
    
    # Use a simpler prompt for connection testing
    test_prompt = "Hi"
        
    try:
        # First check if the API endpoint is reachable
        try:
            health_check = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
            if health_check.status_code != 200:
                logger.warning(f"Ollama API health check failed with status code {health_check.status_code}")
                return False, f"Error: Ollama API health check failed with status code {health_check.status_code}"
        except Exception as e:
            logger.error(f"Ollama API health check failed: {str(e)}")
            return False, f"Error: Cannot reach Ollama API: {str(e)}"
        
        # Now test the model with a simple prompt
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": test_prompt,
                "stream": False
            },
            timeout=timeout
        )
        
        if response.status_code == 200:
            response_text = response.json().get("response", "")
            logger.info(f"Test response: {response_text[:50]}...")
            return True, response_text
        else:
            logger.warning(f"Ollama API returned status code {response.status_code}")
            return False, f"Error: Ollama API returned status code {response.status_code}"
    except requests.exceptions.Timeout:
        logger.error(f"Timeout calling Ollama generate API")
        return False, ""
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error calling Ollama generate API")
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

# Load document embeddings - simplified to avoid parsing errors
def load_document_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            # Create a simplified dictionary with just the embeddings
            embeddings = {}
            for key, value in embeddings_data.items():
                try:
                    # Try to convert the key to a tuple if it looks like one
                    if '(' in key and ')' in key and ',' in key:
                        # This is a simplified approach to handle tuples as keys
                        # We'll just use the document name as the key instead
                        doc_name = key.split(',')[0].strip("('")
                        embeddings[doc_name] = np.array(value)
                    else:
                        # Otherwise just use the key as is
                        embeddings[key] = np.array(value)
                except Exception as key_error:
                    logger.warning(f"Skipping embedding with problematic key: {key}")
                    continue
            
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
def find_relevant_documents(query, top_k=3, max_docs=5):
    # Use the higher of top_k or max_docs
    limit = max(top_k, max_docs)
    logger.info(f"Finding relevant documents with limit={limit}")
    
    # If we don't have embeddings, fall back to keyword search
    if not document_embeddings:
        logger.warning("No embeddings available, falling back to keyword search")
        return find_documents_by_keywords(query, limit)
    
    start_time = time.time()
    query_embedding = get_embedding(query)
    if query_embedding is None:
        logger.warning("Failed to get embedding for query, falling back to keyword search")
        return find_documents_by_keywords(query, limit)
    
    logger.info(f"Generated query embedding in {time.time() - start_time:.2f} seconds")
    
    # Calculate similarity with all documents
    similarities = {}
    calc_start = time.time()
    for doc_id, doc_embedding in document_embeddings.items():
        try:
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities[doc_id] = similarity
        except Exception as e:
            logger.warning(f"Error calculating similarity for document {doc_id}: {str(e)}")
    
    logger.info(f"Calculated similarities for {len(similarities)} documents in {time.time() - calc_start:.2f} seconds")
    
    # Sort by similarity
    sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Get top documents up to the limit
    sort_start = time.time()
    sorted_docs = sorted_docs[:limit]  # Limit to top documents
    logger.info(f"Sorted {len(sorted_docs)} documents in {time.time() - sort_start:.2f} seconds")
    
    # Process the top documents
    relevant_docs = []
    process_start = time.time()
    
    for i, (doc_id, similarity) in enumerate(sorted_docs):
        # Try to find the document in the index
        found = False
        for index_id, doc_info in document_index.items():
            if doc_id in index_id or doc_id == doc_info.get('name', ''):
                doc_info_copy = doc_info.copy()
                doc_info_copy['similarity'] = float(similarity)
                
                # Only include essential content to reduce memory usage
                if 'content' in doc_info_copy and len(doc_info_copy['content']) > 1500:
                    doc_info_copy['content'] = doc_info_copy['content'][:1500] + "... (truncated)"
                    
                relevant_docs.append(doc_info_copy)
                found = True
                break
        
        if not found and i < min(3, len(sorted_docs)):
            logger.warning(f"Document {doc_id} not found in index")
    
    logger.info(f"Processed {len(relevant_docs)} documents in {time.time() - process_start:.2f} seconds")
    
    # If we didn't find any documents with embeddings, fall back to keyword search
    if not relevant_docs:
        logger.warning("No relevant documents found with embeddings, falling back to keyword search")
        return find_documents_by_keywords(query, limit)
    
    return relevant_docs

# Keyword-based document search as fallback
def find_documents_by_keywords(query, top_k=3):
    if not document_index:
        logger.warning("No document index available")
        return []
    
    start_time = time.time()
    logger.info(f"Performing keyword search for: {query[:50]}...")
    
    # Split query into keywords and remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as'}
    keywords = [word.lower() for word in query.split() if word.lower() not in common_words and len(word) > 2]
    
    if not keywords:
        logger.warning("No significant keywords found in query")
        # Use original query words if no significant keywords
        keywords = [word.lower() for word in query.split() if len(word) > 2]
    
    logger.info(f"Using keywords: {keywords}")
    
    # Score documents based on keyword matches
    scores = {}
    for doc_id, doc_info in document_index.items():
        score = 0
        # Check document name (higher weight)
        doc_name = doc_info.get('name', '').lower()
        for keyword in keywords:
            if keyword in doc_name:
                score += 5  # Higher weight for name matches
        
        # Check document path
        doc_path = doc_info.get('simplified_path', '').lower()
        for keyword in keywords:
            if keyword in doc_path:
                score += 2  # Medium weight for path matches
        
        # Check document content if available (sample only for performance)
        doc_content = doc_info.get('content', '')
        if doc_content:
            # Only check the first 5000 characters for performance
            sample = doc_content[:5000].lower()
            for keyword in keywords:
                if keyword in sample:
                    score += 3  # Medium weight for content matches
        
        if score > 0:
            scores[doc_id] = score
    
    logger.info(f"Scored {len(scores)} documents by keywords in {time.time() - start_time:.2f} seconds")
    
    # Sort by score
    sort_start = time.time()
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"Sorted {len(sorted_docs)} documents by score in {time.time() - sort_start:.2f} seconds")
    
    # Get top k documents
    relevant_docs = []
    for i, (doc_id, score) in enumerate(sorted_docs[:top_k]):
        if doc_id in document_index:
            doc_info = document_index[doc_id].copy()
            doc_info['score'] = score
            
            # Truncate content to save memory
            if 'content' in doc_info and len(doc_info['content']) > 1500:
                doc_info['content'] = doc_info['content'][:1500] + "... (truncated)"
                
            relevant_docs.append(doc_info)
    
    logger.info(f"Returning {len(relevant_docs)} relevant documents from keyword search")
    return relevant_docs

# Update the model based on user selection
def update_model(model_display_name):
    global CURRENT_MODEL, OLLAMA_AVAILABLE, OLLAMA_WORKING
    
    # First show loading indicator
    status_text = f"Switching to {model_display_name} model... (this may take up to 60 seconds)"
    status_class = "status-loading"
    
    # Get the actual model name from the display name
    if model_display_name in AVAILABLE_MODELS:
        new_model = AVAILABLE_MODELS[model_display_name]
        logger.info(f"Changing model from {CURRENT_MODEL} to {new_model}")
        
        # Return loading indicator first
        yield status_text, status_class
        
        # Perform a more thorough check for the new model
        try:
            # First check if API is reachable at all
            try:
                health_check = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
                if health_check.status_code != 200:
                    logger.error(f"Ollama API health check failed with status code {health_check.status_code}")
                    OLLAMA_AVAILABLE = False
                    OLLAMA_WORKING = False
                    return f"Cannot connect to GPU machine at {OLLAMA_API_URL}", "status-disconnected"
            except Exception as e:
                logger.error(f"Ollama API health check failed: {str(e)}")
                OLLAMA_AVAILABLE = False
                OLLAMA_WORKING = False
                return f"Cannot connect to GPU machine: {str(e)}", "status-disconnected"
            
            # Now check if the specific model is available
            OLLAMA_AVAILABLE = check_ollama_availability(new_model)
            if OLLAMA_AVAILABLE:
                # Only update the current model if it's available
                CURRENT_MODEL = new_model
                OLLAMA_WORKING, test_response = test_ollama_connection(new_model)
                
                if OLLAMA_WORKING:
                    status_text = f"Model changed to {model_display_name} - Connected to GPU"
                    status_class = "status-connected"
                else:
                    status_text = f"Model found but not responding: {model_display_name}"
                    status_class = "status-disconnected"
            else:
                status_text = f"Model {model_display_name} not found on GPU server"
                status_class = "status-disconnected"
                OLLAMA_WORKING = False
        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            status_text = f"Error updating model: {str(e)}"
            status_class = "status-disconnected"
            OLLAMA_AVAILABLE = False
            OLLAMA_WORKING = False
        
        return status_text, status_class
    else:
        logger.error(f"Unknown model: {model_display_name}")
        return f"Error: Unknown model {model_display_name}", "status-disconnected"

# Generate response from Ollama
def generate_response(prompt, system_prompt=None):
    try:
        # Prepare the request
        request_data = {
            "model": CURRENT_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_data["system"] = system_prompt
        
        # Set a longer timeout for all models
        timeout = 180  # 3 minutes timeout for all models
        logger.info(f"Generating response using model {CURRENT_MODEL} with timeout {timeout}s")
        
        # Send the request with timeout
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=request_data,
            timeout=timeout  # Adjusted timeout based on model
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            response_text = response_json.get("response", "")
            return response_text
        else:
            logger.error(f"Ollama API returned status code {response.status_code}")
            return f"Error: Ollama API returned status code {response.status_code}"
    
    except requests.exceptions.Timeout:
        logger.error("Timeout generating response from Ollama API")
        return "Error: Request to Ollama API timed out. The model might be busy processing another request."
    except requests.exceptions.ConnectionError:
        logger.error("Connection error generating response from Ollama API")
        return "Error: Could not connect to Ollama API. Please check if the server is running."
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

# Process user question
def process_question(message, history, doc_paths_state):
    # Check if connection is stale and needs to be refreshed
    global LAST_CONNECTION_CHECK, OLLAMA_AVAILABLE, OLLAMA_WORKING
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
    
    # Check if Ollama is available before proceeding
    if not OLLAMA_AVAILABLE or not OLLAMA_WORKING:
        logger.error("Ollama API is not available or not working")
        error_message = "Error: Cannot connect to the GPU machine running Ollama. Please check the connection and try again."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_message})
        return history, doc_paths_state, ""
    
    # First add user message to history with a loading indicator for the assistant
    loading_message = "*Generating response...*"
    if "mixtral" in CURRENT_MODEL.lower():
        loading_message = "*Generating response using Mixtral (this may take up to 2 minutes)...*"
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": loading_message})
    yield history, doc_paths_state, ""
    
    try:
        # Get relevant documents with a limit to avoid processing too many
        try:
            # Log that we're starting document retrieval
            logger.info(f"Finding relevant documents for: {message[:50]}...")
            relevant_docs = find_relevant_documents(message, max_docs=5)  # Limit to top 5 docs
            logger.info(f"Found {len(relevant_docs)} relevant documents")
        except Exception as doc_error:
            logger.error(f"Error finding relevant documents: {str(doc_error)}")
            relevant_docs = []
        
        # Update document paths state
        doc_paths_state = [doc["path"] for doc in relevant_docs]
        
        # Create context from relevant documents
        context = ""
        for i, doc in enumerate(relevant_docs):
            # Limit content size to avoid context overflow
            content = doc.get('content', 'No content available')
            if content and len(content) > 1000:
                content = content[:1000] + "... (content truncated)"
                
            context += f"Document {i+1}: {doc['name']} (in {doc['simplified_path']})\n"
            context += f"Content: {content}\n\n"
        
        # Create prompt with context - make it more concise
        prompt = f"""User question about Alstom project: {message}

Relevant document excerpts:
{context}

Provide a concise answer based on these documents. If information is missing, say so briefly."""
        
        # Add a system prompt to guide the model
        system_prompt = """You are the Alstom Project Assistant, a helpful AI that provides concise, accurate information about the Alstom project based on the provided documents. 
- Keep responses brief and to the point
- Focus only on information in the provided documents
- If you're not sure, say so rather than making things up
- Respond within 30 seconds"""
        
        
        # Generate response
        try:
            response_text = generate_response(prompt, system_prompt)
            
            # Create document links HTML
            doc_links_html = ""
            if relevant_docs:
                doc_links_html = "<div style='margin-top: 10px;'><strong>Referenced Documents:</strong></div>"
                for doc in relevant_docs:
                    doc_path = doc["path"]
                    doc_name = doc["name"]
                    doc_links_html += f"<div class='document-link'><a href='http://localhost:{DOC_SERVER_PORT}?path={urllib.parse.quote(doc_path)}' target='_blank'>{doc_name}</a></div>"
            
            # Update history with actual response
            # Remove the loading message
            history.pop()
            # Add the actual response
            history.append({"role": "assistant", "content": response_text})
            
            return history, doc_paths_state, doc_links_html
            
        except requests.exceptions.Timeout:
            error_message = f"Error: The {CURRENT_MODEL} model is taking too long to respond. This might happen with larger models like Mixtral. Please try again or switch to a smaller model like Mistral."
            # Remove the loading message
            history.pop()
            # Add the error message
            history.append({"role": "assistant", "content": error_message})
            return history, doc_paths_state, ""
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        error_message = "Error: An unexpected error occurred while processing your question. Please try again."
        # If there's a loading message, remove it
        if history and history[-1]["role"] == "assistant":
            history.pop()
        # Add the error message
        history.append({"role": "assistant", "content": error_message})
        return history, doc_paths_state, ""
        # Create HTML for document buttons with multiple access options
        doc_buttons_html = "<div style='margin-top: 10px;'><strong>Document References:</strong><br>"
        
        # Get the local machine's IP address for the document server
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except:
            local_ip = "localhost"
        
        for i, doc in enumerate(relevant_docs):
            doc_path = doc['local_path'] if 'local_path' in doc and doc['local_path'] else doc['path']
            doc_name = doc['name']
            simplified_path = doc['simplified_path']
            
            # Convert local path to network path
            network_path = doc_path.replace("C:\\alstom", "\\\\fileserver2\\Head Office Server\\Projects Control (PC)\\10 Backup\\05 Models\\alstom")
            
            # Create URL for document server
            doc_server_url = f"http://{local_ip}:{DOC_SERVER_PORT}/document?path={urllib.parse.quote(network_path)}"
            
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
            </ol>
        </div>
        </div>'''
        
        # Add the new message pair to history
        new_history = history.copy()
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": formatted_response})
        return new_history, doc_paths, doc_buttons_html
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        new_history = history.copy()
        new_history.append({"role": "user", "content": message})
        new_history.append({"role": "assistant", "content": f"Error processing your question: {str(e)}"})
        return new_history, doc_paths_state, ""

# Find an available port
def find_available_port(start_port=7860, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except socket.error:
                continue
    return None

def main():
    global OLLAMA_AVAILABLE, OLLAMA_WORKING, EMBEDDING_MODEL, document_index, document_embeddings, CHATBOT_PORT
    
    # Start document server
    doc_server_started = start_document_server()
    if not doc_server_started:
        logger.warning("Document server could not be started. Document access may be limited.")
    
    # Check Ollama availability
    logger.info(f"Checking connection to Ollama API at {OLLAMA_API_URL}...")
    OLLAMA_AVAILABLE = check_ollama_availability()
    
    if OLLAMA_AVAILABLE:
        logger.info("Successfully connected to Ollama API and found required model")
        
        # Test Ollama connection
        logger.info("Testing Ollama with a simple prompt...")
        OLLAMA_WORKING, test_response = test_ollama_connection()
        
        if OLLAMA_WORKING:
            logger.info("Successfully tested Ollama generate API")
            logger.info(f"Test response: {test_response[:100]}...")  # Log first 100 chars of response
        else:
            logger.error("Failed to test Ollama generate API")
    else:
        logger.error("Failed to connect to Ollama API or model not found")
    
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    EMBEDDING_MODEL = initialize_embedding_model()
    
    # Load document index
    logger.info("Loading document index...")
    document_index = load_document_index()
    
    # Load document embeddings
    logger.info("Loading document embeddings...")
    document_embeddings = load_document_embeddings()
    
    # Find an available port for the chatbot
    CHATBOT_PORT = find_available_port()
    if CHATBOT_PORT is None:
        logger.error("Could not find an available port. Exiting.")
        return
    
    logger.info(f"Starting chatbot on port {CHATBOT_PORT}")
    
    # Create the Gradio interface
    with gr.Blocks(css="""
        footer {visibility: hidden}
        .left-panel { border-right: 1px solid #e0e0e0; }
        .status-connected { background-color: #d4edda; color: #155724; border-radius: 5px; }
        .status-disconnected { background-color: #f8d7da; color: #721c24; border-radius: 5px; }
        .status-loading { background-color: #fff3cd; color: #856404; border-radius: 5px; }
        .document-link { margin-bottom: 5px; padding: 5px; border-radius: 5px; background-color: #e9ecef; }
        .document-link a { color: #0366d6; text-decoration: none; }
        .document-link a:hover { text-decoration: underline; }
        .model-info { font-size: 0.9em; margin-top: 10px; }
        .model-info ul { padding-left: 20px; }
    """) as demo:
        # Store document paths for the current conversation
        doc_paths = gr.State([])
        
        with gr.Row():
            # Left panel (1/4 of screen)
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                # Company logo at the top followed by Alstom logo
                with gr.Column(elem_classes=["logo-container"]):
                    try:
                        gr.Image(os.path.join(os.path.dirname(__file__), "static", "rme.png"), 
                                show_label=False, 
                                height=80)
                        gr.Image(os.path.join(os.path.dirname(__file__), "static", "logo.svg"), 
                                show_label=False, 
                                height=60)
                    except:
                        gr.Markdown("## RME - Alstom")
                
                # GPU Status indicator
                if OLLAMA_AVAILABLE and OLLAMA_WORKING:
                    status_text = "Connected to NVIDIA 4090 GPU"
                    status_class = "status-connected"
                else:
                    status_text = "Cannot connect to GPU machine"
                    status_class = "status-disconnected"
                
                status_html = gr.HTML(f"<div class='{status_class}' style='margin-top: 10px; padding: 10px;'>{status_text}</div>")
                
                # Model status display
                model_status = gr.Markdown(f"**Current Model:** {CURRENT_MODEL}")
                model_info = gr.Markdown("""
                - **Mistral (7B)**: Faster responses, good for most questions
                - **Mixtral (46.7B)**: More accurate but slower, better for complex questions
                """)
                
                
                # Document stats
                gr.Markdown(f"Documents: {len(document_index)}")
                
                # Document server status
                if doc_server_started:
                    gr.Markdown(f"Document Server: Running on port {DOC_SERVER_PORT}")
                else:
                    gr.Markdown("Document Server: Not running")
                
                # Ollama connection details
                gr.Markdown(f"Ollama API: {OLLAMA_API_URL}")
                
                # Model selection dropdown
                gr.Markdown("### Model Selection")
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=list(AVAILABLE_MODELS.keys())[0],  # Default to first model (Mistral)
                    label="Select AI Model",
                    info="Choose between Mistral (7B) and Mixtral (46.7B) models"
                )
                
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
                
                # Chat interface with type specified to avoid deprecation warning
                chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=500, bubble_full_width=False, show_copy_button=True,
                    [],
                    elem_id="chatbot",
                    height=500,
                    type="messages"
                )
                
                # Input area - moved up before document buttons for better UX
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about the Alstom project...",
                        show_label=False,
                        container=False,
                        scale=4  # Takes 80% of the width
                    )
                    submit = gr.Button(
                        "Send",
                        scale=1,  # Takes 20% of the width
                        size="sm"  # Smaller button
                    )
                
                # Document buttons container
                doc_buttons = gr.HTML()
                
                # Set up event handlers
                submit_event = msg.submit(
                    process_question, 
                    [msg, chatbot, doc_paths], 
                    [chatbot, doc_paths, doc_buttons]
                )
                submit_click_event = submit.click(
                    process_question, 
                    [msg, chatbot, doc_paths], 
                    [chatbot, doc_paths, doc_buttons]
                )
                
                # Clear the input box after submission
                submit_event.then(lambda: "", None, msg)
                submit_click_event.then(lambda: "", None, msg)
                
                # Add event handler for model selection
                model_dropdown.change(
                    fn=update_model,
                    inputs=[model_dropdown],
                    outputs=[status_html, model_status],
                    show_progress=True  # Show progress indicator during model switching
                ).then(
                    fn=lambda model_name: f"**Current Model:** {AVAILABLE_MODELS[model_name]}",
                    inputs=[model_dropdown],
                    outputs=[model_status]
                )
    
    # Launch the interface with the dynamically assigned port
    demo.launch(server_name="0.0.0.0", server_port=CHATBOT_PORT)

if __name__ == "__main__":
    main()
