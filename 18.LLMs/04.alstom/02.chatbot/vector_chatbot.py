"""
Alstom Project Assistant - Vector-based Chatbot Interface
This script provides a web interface for the Alstom Project Assistant.
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
            model_names = [model["name"] for model in response.json().get("models", [])]
            if MODEL_NAME in model_names:
                logger.info(f"Found {MODEL_NAME} model on the server")
                return True
            else:
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
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": "Are you there?", "stream": False},
            timeout=10
        )
        if response.status_code == 200:
            response_text = response.json().get("response", "")
            logger.info(f"Test response: {response_text[:50]}...")
            return True, response_text
        else:
            logger.warning(f"Ollama generate API returned status code {response.status_code}")
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
        if not OLLAMA_AVAILABLE or not OLLAMA_WORKING:
            return f"<div style='color: red;'>Cannot connect to the GPU machine. Please check the connection and try again.</div>", []
        
        # Find relevant documents
        relevant_docs = find_relevant_documents(message)
        doc_paths = []
        
        # Create context from relevant documents
        context = "Based on the following documents:\n\n"
        for i, doc in enumerate(relevant_docs):
            doc_ref = f"Document: {doc['name']} (in {doc['simplified_path']})"
            context += f"{i+1}. {doc_ref}\n"
            doc_paths.append(doc['local_path'] if doc['local_path'] else doc['path'])
        
        context += "\nPlease answer the following question: " + message
        
        # Generate response using Ollama
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": context, "stream": False},
            timeout=30
        )
        
        if response.status_code != 200:
            return f"Error: Ollama API returned status code {response.status_code}", []
        
        # Extract the response text
        response_text = response.json().get("response", "")
        
        # Format the response with document sources
        formatted_response = response_text + "\n\nSources:\n"
        for i, doc in enumerate(relevant_docs):
            formatted_response += f"- {doc['name']} (in {doc['simplified_path']})\n"
        
        # Create HTML for document buttons
        doc_buttons_html = "<div style='margin-top: 10px;'><strong>Open Documents:</strong><br>"
        for i, doc in enumerate(relevant_docs):
            doc_path = doc['local_path'] if doc['local_path'] else doc['path']
            doc_name = doc['name']
            doc_buttons_html += f"<button onclick=\"window.open('file:///{doc_path.replace('\\', '/')}', '_blank')\" style='margin: 5px; padding: 5px 10px; background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 4px; cursor: pointer;'>{doc_name}</button>"
        doc_buttons_html += "</div>"
        
        return formatted_response, doc_paths, doc_buttons_html
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return f"I'm sorry, there was an error processing your question: {str(e)}", [], ""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(css="""
        footer {visibility: hidden}
        .left-panel { border-right: 1px solid #e0e0e0; }
        .status-connected { color: green; font-weight: bold; }
        .status-disconnected { color: red; font-weight: bold; }
    """) as demo:
        # Store document paths for the current conversation
        doc_paths = gr.State([])
        
        with gr.Row():
            # Left panel (1/4 of screen)
            with gr.Column(scale=1, elem_classes=["left-panel"]):
                # Logo at the top
                gr.Image("static/logo.png", width=150, show_label=False)
                
                # GPU Status indicator
                if OLLAMA_AVAILABLE and OLLAMA_WORKING:
                    status_class = "status-connected"
                    status_text = "✓ Connected to NVIDIA 4090 GPU"
                else:
                    status_class = "status-disconnected"
                    status_text = "✗ Cannot connect to GPU machine"
                    
                gr.HTML(f"<div class='{status_class}' style='margin-top: 10px; padding: 10px;'>{status_text}</div>")
                
                # Document stats
                gr.HTML(f"<div style='margin-top: 20px; padding: 10px;'><strong>Documents:</strong> {len(document_index)}</div>")
                
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
                chatbot = gr.Chatbot(
                    show_label=False,
                    bubble_full_width=False,
                    height=500,  # Increased height slightly
                    value=[["", "Welcome to the Alstom Project Assistant! I can answer questions about the project documents. How can I help you today?"]]
                )
                
                # Document buttons area
                doc_buttons = gr.HTML("")
                
                # Input area always at the bottom
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
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(share=False)

if __name__ == "__main__":
    main()
