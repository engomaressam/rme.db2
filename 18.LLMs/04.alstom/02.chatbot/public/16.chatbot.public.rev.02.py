"""
Alstom Project Assistant - Public Chatbot Interface
This script provides a web interface for the Alstom Project Assistant with public sharing enabled.
It uses pre-processed document embeddings to retrieve relevant documents.
"""

import os
import json
import numpy as np
import requests
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
import time
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("public_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PublicChatbot")

# Define paths
# Note: The vector database remains in the same location, but the documents are now stored on the network
# at \\fileserver2\Head Office Server\Projects Control (PC)\10 Backup\05 Models\alstom
VECTOR_DB_DIR = "C:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\05.llm\\alstom"
DOCUMENTS_DIR = os.path.join(VECTOR_DB_DIR, "documents")
EMBEDDINGS_FILE = os.path.join(VECTOR_DB_DIR, "document_embeddings.json")
DOCUMENTS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "document_index.json")

# Ollama API configuration
OLLAMA_API_URL = "http://10.10.12.202:11434"  # IP address of the GPU machine
MODEL_NAME = "mistral"  # Model to use for chat

# Global variables
OLLAMA_AVAILABLE = False
OLLAMA_WORKING = False
EMBEDDING_MODEL = None
document_index = {}
document_embeddings = {}

# Function to check if Ollama API is available
def check_ollama_availability():
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            # Check if our model is available
            data = response.json()
            model_names = []
            
            # Handle different API response formats
            if "models" in data:
                model_names = [model["name"] for model in data["models"]]
            elif "models" not in data and isinstance(data, list):
                # Newer Ollama API returns a list directly
                model_names = [model["name"] for model in data]
            
            logger.info(f"Available models on server: {', '.join(model_names)}")
            
            if MODEL_NAME in model_names:
                logger.info(f"Found {MODEL_NAME} model on the server")
                return True
            else:
                # Try with partial match (e.g., 'mistral' might be 'mistral:latest')
                for model in model_names:
                    if MODEL_NAME in model:
                        logger.info(f"Found {model} which contains {MODEL_NAME}")
                        return True
                        
                logger.warning(f"Model {MODEL_NAME} not found on server. Available models: {', '.join(model_names)}")
                return False
        else:
            logger.warning(f"Ollama API returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking Ollama availability: {str(e)}")
        return False

# Function to test the Ollama connection with a simple prompt
def test_ollama_connection():
    try:
        # First try with the exact model name
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": "Are you there?", "stream": False},
                timeout=10
            )
            if response.status_code == 200:
                response_text = response.json().get("response", "")
                logger.info(f"Test response: {response_text[:50]}...")
                return True, response_text
        except Exception as model_error:
            logger.warning(f"Error with exact model name: {str(model_error)}")
            
            # Try to get available models and use the first one that contains our model name
            try:
                models_response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
                if models_response.status_code == 200:
                    data = models_response.json()
                    model_names = []
                    
                    # Handle different API response formats
                    if "models" in data:
                        model_names = [model["name"] for model in data["models"]]
                    elif "models" not in data and isinstance(data, list):
                        model_names = [model["name"] for model in data]
                    
                    # Find a model that contains our model name
                    for model in model_names:
                        if MODEL_NAME in model:
                            logger.info(f"Trying with model: {model}")
                            response = requests.post(
                                f"{OLLAMA_API_URL}/api/generate",
                                json={"model": model, "prompt": "Are you there?", "stream": False},
                                timeout=10
                            )
                            if response.status_code == 200:
                                response_text = response.json().get("response", "")
                                logger.info(f"Test response with {model}: {response_text[:50]}...")
                                return True, response_text
            except Exception as list_error:
                logger.error(f"Error getting model list: {str(list_error)}")
        
        logger.warning(f"All attempts to connect to Ollama failed")
        return False, None
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {str(e)}")
        return False, None

# Initialize embedding model
def initialize_embedding_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded SentenceTransformer model")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        return None

# Load document index
def load_document_index():
    if not os.path.exists(DOCUMENTS_INDEX_FILE):
        logger.warning(f"Document index file not found: {DOCUMENTS_INDEX_FILE}")
        return {}
    
    try:
        with open(DOCUMENTS_INDEX_FILE, 'r', encoding='utf-8') as f:
            index = json.load(f)
        logger.info(f"Loaded document index with {len(index)} documents")
        return index
    except Exception as e:
        logger.error(f"Error loading document index: {str(e)}")
        return {}

# Load document embeddings
def load_document_embeddings():
    if not os.path.exists(EMBEDDINGS_FILE):
        logger.warning(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        return {}
    
    try:
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        
        # Convert lists back to numpy arrays
        embeddings_np = {doc_id: np.array(emb) for doc_id, emb in embeddings.items()}
        logger.info(f"Loaded embeddings for {len(embeddings_np)} documents")
        return embeddings_np
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        return {}

# Find relevant documents using vector similarity
def find_relevant_documents(query, top_n=3):
    if not document_embeddings or EMBEDDING_MODEL is None:
        logger.warning("Cannot find relevant documents: embeddings or model not loaded")
        return []
    
    try:
        # Encode the query
        query_embedding = EMBEDDING_MODEL.encode(query)
        
        # Calculate similarity scores
        similarities = {}
        for doc_id, doc_embedding in document_embeddings.items():
            similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarities[doc_id] = similarity
        
        # Sort by similarity score
        sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N documents
        top_docs = []
        for doc_id, score in sorted_docs[:top_n]:
            if doc_id in document_index:
                doc_info = document_index[doc_id]
                top_docs.append({
                    "name": doc_info["name"],
                    "path": doc_info["path"],
                    "simplified_path": doc_info["simplified_path"],
                    "local_path": doc_info.get("local_path", ""),
                    "score": score
                })
        
        logger.info(f"Found {len(top_docs)} relevant documents for query: {query}")
        return top_docs
    except Exception as e:
        logger.error(f"Error finding relevant documents: {str(e)}")
        return []

# Process user question
def process_question(message, history, doc_paths_state):
    try:
        # Check if message is empty
        if not message or message.strip() == "":
            return history, doc_paths_state, ""
            
        # Check if Ollama is available
        if not OLLAMA_AVAILABLE or not OLLAMA_WORKING:
            new_history = history.copy()
            new_history.append([message, f"<div style='color: red;'>Cannot connect to the GPU machine. Please check the connection and try again.</div>"])
            return new_history, doc_paths_state, ""
        
        # Find relevant documents
        relevant_docs = find_relevant_documents(message)
        doc_paths = doc_paths_state.copy() if doc_paths_state else []
        
        # Handle case where no documents are processed yet
        if not document_embeddings or len(document_embeddings) == 0:
            new_history = history.copy()
            new_history.append([message, "The document processor is still indexing documents. Please try again later when some documents have been processed."])
            return new_history, doc_paths, ""
        
        # If no relevant documents found
        if not relevant_docs:
            new_history = history.copy()
            new_history.append([message, "I couldn't find any relevant documents for your query. The document processor might still be indexing documents or your query might be about a topic not covered in the available documents."])
            return new_history, doc_paths, ""
        
        # Create context from relevant documents
        context = "Based on the following documents:\n\n"
        for i, doc in enumerate(relevant_docs):
            doc_ref = f"Document: {doc['name']} (in {doc['simplified_path']})"
            context += f"{i+1}. {doc_ref}\n"
            doc_paths.append(doc['local_path'] if doc['local_path'] else doc['path'])
        
        context += "\nPlease answer the following question: " + message
        
        # Generate response using Ollama
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": context, "stream": False},
                timeout=30
            )
            
            if response.status_code != 200:
                new_history = history.copy()
                new_history.append([message, f"Error: Ollama API returned status code {response.status_code}"])
                return new_history, doc_paths, ""
            
            # Extract the response text
            response_text = response.json().get("response", "")
        except Exception as api_error:
            logger.error(f"Error calling Ollama API: {str(api_error)}")
            new_history = history.copy()
            new_history.append([message, f"Error connecting to Ollama API: {str(api_error)}"])
            return new_history, doc_paths, ""
        
        # Format the response with document sources
        formatted_response = response_text + "\n\nSources:\n"
        for i, doc in enumerate(relevant_docs):
            formatted_response += f"- {doc['name']} (in {doc['simplified_path']})\n"
        
        # Create HTML for document info (no buttons since files are local)
        doc_info_html = "<div style='margin-top: 10px;'><strong>Document Sources:</strong><br>"
        for i, doc in enumerate(relevant_docs):
            doc_name = doc['name']
            doc_path = doc['simplified_path']
            doc_info_html += f"<div style='margin: 5px; padding: 5px 10px; background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 4px;'>{doc_name} (in {doc_path})</div>"
        doc_info_html += "</div>"
        
        # Add the new message pair to history
        new_history = history.copy()
        new_history.append([message, formatted_response])
        return new_history, doc_paths, doc_info_html
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        new_history = history.copy()
        new_history.append([message, f"I'm sorry, there was an error processing your question: {str(e)}"])
        return new_history, doc_paths_state, ""

# Encode images to base64 for web display
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            file_extension = os.path.splitext(image_path)[1].lower()
            if file_extension == '.svg':
                mime_type = 'image/svg+xml'
            elif file_extension == '.png':
                mime_type = 'image/png'
            else:
                mime_type = 'image/jpeg'
            return f"data:{mime_type};base64,{encoded_string}"
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        return ""

# Create the Gradio interface
def create_interface():
    # Get base64 encoded images for logos
    rme_logo_path = os.path.join(os.path.dirname(__file__), "static", "rme.png")
    alstom_logo_path = os.path.join(os.path.dirname(__file__), "static", "logo.svg")
    
    rme_logo_base64 = get_base64_image(rme_logo_path)
    alstom_logo_base64 = get_base64_image(alstom_logo_path)
    
    with gr.Blocks(css="""
        footer {visibility: hidden}
        .left-panel { border-right: 1px solid #e0e0e0; }
        .status-connected { color: green; font-weight: bold; }
        .status-disconnected { color: red; font-weight: bold; }
        .logo-container { text-align: center; margin-bottom: 15px; }
        .logo-container img { max-width: 150px; margin-bottom: 10px; }
    """) as demo:
        # Store document paths for the current conversation
        doc_paths = gr.State([])
        
        with gr.Row():
            # Left panel (1/4 of screen)
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                # Company logo at the top followed by Alstom logo
                if rme_logo_base64:
                    gr.HTML(f"""<div class="logo-container"><img src="{rme_logo_base64}" alt="RME Logo"></div>""")
                if alstom_logo_base64:
                    gr.HTML(f"""<div class="logo-container"><img src="{alstom_logo_base64}" alt="Alstom Logo"></div>""")
                
                # GPU Status indicator
                if OLLAMA_AVAILABLE and OLLAMA_WORKING:
                    status_class = "status-connected"
                    status_text = "✓ Connected to NVIDIA 4090 GPU"
                else:
                    status_class = "status-disconnected"
                    status_text = "✗ Cannot connect to GPU machine"
                    
                gr.HTML(f"<div class='{status_class}' style='margin-top: 10px; padding: 10px;'>{status_text}</div>")
                
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
                
                # Chat area with fixed height to avoid scrolling
                with gr.Column(elem_classes=["chat-container"]):
                    chatbot = gr.Chatbot(
                        show_label=False,
                        bubble_full_width=False,
                        height=350,  # Reduced height to fit input box below
                        value=[["", "Welcome to the Alstom Project Assistant! I can answer questions about the project documents. How can I help you today?"]]
                    )
                    
                    # Document buttons area
                    doc_buttons = gr.HTML("")
                    
                    # Input area below the chat for better user experience
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask a question about the Alstom project...",
                            show_label=False,
                            scale=9,
                        )
                        submit = gr.Button("Send", scale=1)
        
        # Set up event handlers
        submit.click(
            process_question,
            [msg, chatbot, doc_paths],
            [chatbot, doc_paths, doc_buttons],
            queue=False
        ).then(
            lambda: "",
            None,
            [msg],
            queue=False
        )
        
        msg.submit(
            process_question,
            [msg, chatbot, doc_paths],
            [chatbot, doc_paths, doc_buttons],
            queue=False
        ).then(
            lambda: "",
            None,
            [msg],
            queue=False
        )
    
    return demo

# Main function
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
    
    print("\n" + "=" * 80)
    print("LAUNCHING PUBLIC CHATBOT WITH GRADIO SHARING")
    print("=" * 80)
    print("\nNOTE: For Gradio public sharing to work, your IT department needs to allow:")
    print("1. Outbound HTTPS connections to *.gradio.live and *.gradio.app domains")
    print("2. Outbound connections to port 443 (HTTPS)")
    print("3. Inbound connections to your local port 7860")
    print("\nIf the sharing link doesn't appear, please contact your IT department.")
    print("=" * 80 + "\n")
    
    # Configure Gradio for public sharing
    demo.queue()
    
    # Launch with sharing enabled
    # This will create a public URL like https://xxxx.gradio.app
    # The URL will be valid for 72 hours
    print("Generating public sharing link...")
    print("Look for 'Running on public URL: https://xxxx.gradio.app' in the output below")
    print("\n")
    
    # Launch with sharing enabled
    demo.launch(share=True)
    
    # The script will block here until terminated

if __name__ == "__main__":
    main()
