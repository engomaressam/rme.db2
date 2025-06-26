import os
import json
import gradio as gr
import requests
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime

# Configuration
OLLAMA_API_URL = "http://10.10.12.202:11434"  # GPU machine's IP
MODEL_NAME = "mistral"  # Using mistral model which is available on the server

# Check if Ollama API is available and if the model is loaded
def check_ollama_availability():
    try:
        # First check if the API is responding
        version_response = requests.get(f"{OLLAMA_API_URL}/api/version", timeout=5)
        if version_response.status_code != 200:
            print(f"Ollama API returned status code: {version_response.status_code}")
            return False
            
        # Then check if our model is available
        tags_response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if tags_response.status_code != 200:
            print(f"Ollama tags API returned status code: {tags_response.status_code}")
            return False
            
        # Parse the response to check for our model
        models = tags_response.json().get('models', [])
        model_names = [model.get('name').split(':')[0] for model in models]
        
        if MODEL_NAME in model_names:
            print(f"Found {MODEL_NAME} model on the server")
            return True
        else:
            print(f"Model {MODEL_NAME} not found on server. Available models: {', '.join(model_names)}")
            return False
    except Exception as e:
        print(f"Error checking Ollama availability: {str(e)}")
        return False

print(f"Checking connection to Ollama API at {OLLAMA_API_URL}...")
OLLAMA_AVAILABLE = check_ollama_availability()
if OLLAMA_AVAILABLE:
    print("✓ Successfully connected to Ollama API and found required model")
else:
    print("✗ Could not connect to Ollama API or model not available - will use fallback responses")

# Load summaries for quick access
def load_summaries():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    summaries_path = os.path.join(data_dir, "ollama_summaries.json")
    
    if not os.path.exists(summaries_path):
        return {}
    
    try:
        with open(summaries_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

# Load document information
def load_document_info():
    # Try to get actual PDF files from the Alstom directory
    real_base_path = "C:\\alstom"
    pdf_files = []
    
    # Try to scan the actual directory if it exists
    if os.path.exists(real_base_path):
        print(f"Scanning {real_base_path} for PDF files...")
        for root, dirs, files in os.walk(real_base_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    # Create a simplified path relative to the base path
                    rel_path = os.path.relpath(os.path.dirname(full_path), real_base_path)
                    pdf_files.append({
                        "name": file,
                        "path": full_path,
                        "simplified_path": rel_path,
                        "content": extract_pdf_content(full_path)  # Extract text content from PDF
                    })
        print(f"Found {len(pdf_files)} PDF files in {real_base_path}")
    
    # If no files found or directory doesn't exist, use sample data
    if not pdf_files:
        print(f"Using sample document data (could not access {real_base_path})")
        pdf_files = [
            {
                "name": "Steel works specs.pdf", 
                "path": "C:\\alstom\\folder1\\06- CR\\02- CR02 Steel works\\Steel works specs.pdf", 
                "simplified_path": "06- CR\\02- CR02 Steel works",
                "content": "This document provides details for a new manufacturing cabling site project, focusing on structural steel specifications. It includes sections for constituent products, fabrication and welding, mechanical fastening, erection, surface treatment, coating, tolerances, bearing, design, definitions, specifications and documentation, and constituent products."
            },
            {
                "name": "Alstom Factory - Borg el Arab - Roads IFC Drawings 27-3-2025.pdf", 
                "path": "C:\\alstom\\folder1\\01- IFC drawings\\9- Roads Drawings\\Alstom Factory - Borg el Arab - Roads IFC Drawings 27-3-2025.pdf", 
                "simplified_path": "01- IFC drawings\\9- Roads Drawings",
                "content": "These drawings show the road layout and design for the Alstom Factory in Borg el Arab. They include details on road dimensions, materials, drainage systems, traffic signs, and road markings. The drawings are issued for construction (IFC) and dated March 27, 2025."
            },
            {
                "name": "Coordinates Topographic Survey Report.pdf", 
                "path": "C:\\alstom\\folder2\\09- Topo\\01- 2025-05-05\\Coordinates Topographic Survey Report.pdf", 
                "simplified_path": "09- Topo\\01- 2025-05-05",
                "content": "This report contains the topographic survey data for the project site. It includes elevation data, boundary coordinates, existing structures, utilities, and natural features. The survey was conducted on May 5, 2025, and provides the baseline data for site planning and design."
            },
            {
                "name": "Electrical Systems Design.pdf", 
                "path": "C:\\alstom\\folder1\\03- MEP\\01- Electrical\\Electrical Systems Design.pdf", 
                "simplified_path": "03- MEP\\01- Electrical",
                "content": "This document details the electrical systems design for the Alstom project. It covers power distribution, lighting systems, emergency power, grounding, lightning protection, and telecommunications. The design includes specifications for electromechanical components including motors, generators, transformers, and control systems."
            },
            {
                "name": "Mechanical Systems Specifications.pdf", 
                "path": "C:\\alstom\\folder1\\03- MEP\\02- Mechanical\\Mechanical Systems Specifications.pdf", 
                "simplified_path": "03- MEP\\02- Mechanical",
                "content": "This document provides detailed specifications for the mechanical systems in the Alstom project. It includes HVAC systems, plumbing, fire protection, and various electromechanical components such as pumps, fans, compressors, and their control systems. It details materials, installation requirements, testing procedures, and performance criteria."
            }
        ]
    
    return pdf_files

# Function to extract text content from PDF files
def extract_pdf_content(pdf_path):
    try:
        # Try to use PyPDF2 to extract text
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""  # Return empty string if extraction fails

# Function to test the Ollama connection with a simple prompt
def test_ollama_connection():
    try:
        test_prompt = "Hello, are you working?"
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": test_prompt,
                "stream": False
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'response' in result:
                print("✓ Successfully tested Ollama generate API")
                print(f"Test response: {result['response'][:50]}...")
                return True
        
        print(f"✗ Failed to get proper response from Ollama generate API: {response.status_code}")
        return False
    except Exception as e:
        print(f"✗ Error testing Ollama connection: {str(e)}")
        return False

# Test the connection if Ollama is available
if OLLAMA_AVAILABLE:
    print("Testing Ollama with a simple prompt...")
    OLLAMA_WORKING = test_ollama_connection()
else:
    OLLAMA_WORKING = False

# Initialize the sentence transformer model for embeddings
def initialize_embedding_model():
    try:
        # Try to load a pre-trained model
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model good for semantic search
        print("✓ Sentence transformer model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Error loading sentence transformer model: {str(e)}")
        return None

# Load document summaries and information
print("Loading document summaries...")
summaries = load_summaries()
print(f"Loaded {len(summaries)} document summaries")

print("Loading document information...")
documents = load_document_info()
print(f"Loaded information for {len(documents)} documents")

# Initialize embedding model
print("Initializing embedding model...")
EMBEDDING_MODEL = initialize_embedding_model()

# Define the vector database storage path
VECTOR_DB_DIR = "C:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\05.llm\\alstom"

# Function to create document embeddings
def create_document_embeddings():
    if EMBEDDING_MODEL is None:
        print("Cannot create embeddings: model not loaded")
        return {}
    
    # Create the vector database directory if it doesn't exist
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    
    # Store embeddings in the specified location
    embeddings_path = os.path.join(VECTOR_DB_DIR, "document_embeddings.json")
    
    # Check if embeddings already exist
    if os.path.exists(embeddings_path):
        try:
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                # Convert string embeddings back to numpy arrays
                embeddings = {}
                for doc_id, embedding_str in saved_data.items():
                    embeddings[doc_id] = np.array(embedding_str)
                print(f"Loaded embeddings for {len(embeddings)} documents")
                return embeddings
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
    
    # Create new embeddings
    print("Creating new document embeddings...")
    embeddings = {}
    
    for doc in documents:
        # Create a rich text representation of the document
        doc_text = f"Document: {doc['name']}\nPath: {doc['simplified_path']}\n"
        
        # Add document content if available
        if 'content' in doc and doc['content']:
            doc_text += f"Content: {doc['content']}\n"
        
        # Add summary if available
        if doc['name'] in summaries:
            doc_text += f"Summary: {summaries[doc['name']]}\n"
        
        # Generate embedding
        try:
            embedding = EMBEDDING_MODEL.encode(doc_text)
            doc_id = f"{doc['name']}_{doc['simplified_path']}"
            embeddings[doc_id] = embedding
        except Exception as e:
            print(f"Error embedding document {doc['name']}: {str(e)}")
    
    # Save embeddings for future use
    try:
        # Convert numpy arrays to lists for JSON serialization
        embeddings_to_save = {doc_id: emb.tolist() for doc_id, emb in embeddings.items()}
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_to_save, f)
        print(f"Saved embeddings for {len(embeddings)} documents")
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
    
    return embeddings

# Create document embeddings for vector search
document_embeddings = create_document_embeddings()

# Function to generate an answer using Ollama API
def generate_answer(question, context):
    # If Ollama is not available or not working, return an error message
    if not OLLAMA_AVAILABLE or not OLLAMA_WORKING:
        return f"<div style='color: red; font-weight: bold; padding: 10px; border: 1px solid red; border-radius: 5px; margin: 10px 0;'>⚠️ ERROR: Cannot reach NVIDIA 4090 GPU machine. The AI server appears to be offline. Please contact the system administrator.</div>"
    
    prompt = f"""You are an expert assistant for the Alstom project. Answer the following question based on your knowledge of the project documents.
    
    Context from project documents:
    {context}
    
    Question: {question}
    
    Answer the question based on the context provided. If the context doesn't contain relevant information to answer the question, say so.
    """
    
    # Call Ollama API
    try:
        # Add longer timeout and retry mechanism
        max_retries = 2
        retry_count = 0
        connection_error = None
        
        while retry_count < max_retries:
            try:
                print(f"Sending request to Ollama API ({retry_count+1}/{max_retries+1})...")
                response = requests.post(
                    f"{OLLAMA_API_URL}/api/generate",
                    json={
                        "model": MODEL_NAME,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7}
                    },
                    timeout=30  # 30 second timeout
                )
                
                # Print response status for debugging
                print(f"Ollama API response status: {response.status_code}")
                
                response.raise_for_status()
                result = response.json()
                
                if "response" in result:
                    print("Successfully generated response from Ollama")
                    return result["response"]
                else:
                    print(f"Unexpected response format from Ollama: {result.keys()}")
                    raise Exception("Unexpected response format")
                    
            except requests.exceptions.ConnectionError as ce:
                connection_error = ce
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Connection error, retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)  # Wait before retrying
                else:
                    raise
            except Exception as e:
                print(f"Error during API call: {str(e)}")
                # For non-connection errors, don't retry
                raise e
        
        # If we get here, all retries failed
        raise connection_error
    
    except Exception as e:
        print(f"Error calling Ollama API: {str(e)}")
        # Return an error message instead of using the fallback
        return f"<div style='color: red; font-weight: bold; padding: 10px; border: 1px solid red; border-radius: 5px; margin: 10px 0;'>⚠️ ERROR: Failed to connect to NVIDIA 4090 GPU machine. Error: {str(e)[:100]}...</div>"

# Create sample PDF files for testing
def create_sample_pdfs():
    # Create a documents directory if it doesn't exist
    docs_dir = os.path.join(VECTOR_DB_DIR, "documents")
    os.makedirs(docs_dir, exist_ok=True)
    
    # Create sample PDF files if they don't exist
    sample_files = [
        "Steel works specs.pdf",
        "Alstom Factory - Borg el Arab - Roads IFC Drawings 27-3-2025.pdf",
        "Coordinates Topographic Survey Report.pdf"
    ]
    
    for filename in sample_files:
        filepath = os.path.join(docs_dir, filename)
        if not os.path.exists(filepath):
            # Create a simple text file with PDF extension for demo purposes
            with open(filepath, 'w') as f:
                f.write(f"This is a sample file for {filename}\n")
                f.write("It contains placeholder content for demonstration purposes.\n")
            print(f"Created sample file: {filepath}")

# Call this function to ensure we have sample files
create_sample_pdfs()

# Function to open a document
def open_document(file_path):
    try:
        # Use the appropriate command to open the file based on OS
        import subprocess
        import platform
        
        if platform.system() == 'Windows':
            os.startfile(file_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(['open', file_path])
        else:  # Linux
            subprocess.call(['xdg-open', file_path])
            
        return f"Opening: {file_path}"
    except Exception as e:
        return f"Error opening file: {str(e)}"

# Function to extract file paths from document references
def extract_file_paths(doc_ref):
    # Example: "Document: Steel works specs.pdf (in 06- CR\02- CR02 Steel works)"
    try:
        # Extract the filename from the document reference
        if '(in' in doc_ref:
            parts = doc_ref.split('(in')
            filename = parts[0].replace('Document:', '').strip()
        else:
            # If no path info, just extract the filename
            filename = doc_ref.replace('Document:', '').strip()
        
        # Use our sample documents directory in the vector DB location
        docs_dir = os.path.join(VECTOR_DB_DIR, "documents")
        
        # Check for exact match
        filepath = os.path.join(docs_dir, filename)
        if os.path.exists(filepath):
            return filepath
        
        # Try to find a partial match
        for file in os.listdir(docs_dir):
            if filename.lower() in file.lower():
                return os.path.join(docs_dir, file)
        
        # If still not found, create a placeholder file
        placeholder_path = os.path.join(docs_dir, filename)
        with open(placeholder_path, 'w') as f:
            f.write(f"This is a placeholder for {filename}\n")
            f.write("The actual document was not found in the system.\n")
        print(f"Created placeholder file: {placeholder_path}")
        return placeholder_path
            
    except Exception as e:
        print(f"Error extracting file path: {str(e)}")
        return None

# Function to process a question and return an answer
def process_question(message, history, doc_paths_state):
    try:
        # Check if Ollama is available before proceeding
        if not OLLAMA_AVAILABLE or not OLLAMA_WORKING:
            return f"<div style='color: red; font-weight: bold; padding: 10px; border: 1px solid red; border-radius: 5px; margin: 10px 0;'>⚠️ ERROR: Cannot reach NVIDIA 4090 GPU machine. The AI server appears to be offline. Please contact the system administrator.</div>", []
        
        # Use vector search if the embedding model is available
        if EMBEDDING_MODEL is not None:
            # Encode the question
            question_embedding = EMBEDDING_MODEL.encode(message)
            
            # Calculate similarity with all documents
            similarities = {}
            for doc_id, doc_embedding in document_embeddings.items():
                similarity = cosine_similarity(
                    [question_embedding], 
                    [doc_embedding]
                )[0][0]
                similarities[doc_id] = similarity
            
            # Sort by similarity and get top matches
            sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            top_doc_ids = [doc_id for doc_id, _ in sorted_similarities[:5]]
            
            # Map back to document objects
            top_docs = []
            for doc_id in top_doc_ids:
                doc_name, doc_path = doc_id.split('_', 1)
                for doc in documents:
                    if doc['name'] == doc_name and doc['simplified_path'] == doc_path:
                        doc_copy = doc.copy()
                        doc_copy['relevance'] = similarities[doc_id]
                        top_docs.append(doc_copy)
                        break
        else:
            # Fallback to keyword matching if embedding model is not available
            print("Using keyword matching fallback for document retrieval")
            question_keywords = message.lower().split()
            relevant_docs = []
            
            for doc in documents:
                doc_text = doc['name'].lower() + ' ' + doc['simplified_path'].lower()
                
                # Add summary to doc_text if available
                if doc['name'] in summaries:
                    doc_text += ' ' + summaries[doc['name']].lower()
                
                # Check if any keyword from the question is in the document text
                relevance_score = 0
                for keyword in question_keywords:
                    if len(keyword) > 3 and keyword in doc_text:  # Only consider keywords with length > 3
                        relevance_score += 1
                
                if relevance_score > 0:
                    doc_copy = doc.copy()
                    doc_copy['relevance'] = relevance_score
                    relevant_docs.append(doc_copy)
            
            # Sort by relevance and take top 5
            relevant_docs.sort(key=lambda x: x['relevance'], reverse=True)
            top_docs = relevant_docs[:5]
        
        # Create context from relevant documents
        context = ""
        sources = []
        doc_references = []
        
        for doc in top_docs:
            doc_ref = f"Document: {doc['name']} (in {doc['simplified_path']})"
            context += f"\n{doc_ref}\n"
            sources.append(f"{doc['name']} (in {doc['simplified_path']})")
            doc_references.append(doc_ref)
            
            # Add summary if available
            if doc['name'] in summaries:
                context += f"Summary: {summaries[doc['name']]}\n\n"
        
        # If no relevant documents found, provide a generic response
        if not top_docs:
            return "I couldn't find any relevant documents for your question. Could you try rephrasing it or asking about a specific aspect of the Alstom project?", []
        
        # Generate answer
        answer = generate_answer(message, context)
        
        # Check if the answer is an error message (it will contain HTML)
        if "<div style='color: red;" in answer:
            # If it's an error message, return it without adding sources
            return answer, []
        
        # Add sources to the answer
        if sources:
            answer += "\n\nSources:\n" + "\n".join([f"- {source}" for source in sources])
        
        # Create document paths list for buttons
        paths = []
        for doc_ref in doc_references:
            path = extract_file_paths(doc_ref)
            if path:
                paths.append(path)
        
        # Return the answer string and document paths
        return answer, paths
        
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return f"I'm sorry, there was an error processing your question: {str(e)}", []

# Add a .gitignore file to exclude vector database from git
def create_gitignore():
    gitignore_path = os.path.join(VECTOR_DB_DIR, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write("# Ignore vector database files\n")
            f.write("document_embeddings.json\n")
            f.write("# Ignore document files\n")
            f.write("documents/\n")
        print(f"Created .gitignore file at {gitignore_path}")

# Create .gitignore file
create_gitignore()

# Create the Gradio interface
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
            
            # File download component (hidden)
            file_output = gr.File(label="Document", visible=False)
            
            # Input area always at the bottom
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question about the Alstom project...",
                    show_label=False,
                    scale=9,
                )
                submit = gr.Button("Send", scale=1)
    
    # Function to serve a file for download
    def serve_file(file_path):
        if file_path and os.path.exists(file_path):
            return file_path
        return None
    
    # Create file download buttons
    def create_file_buttons(paths):
        buttons_html = ""
        valid_paths = []
        
        if paths:
            for path in paths:
                if path and os.path.exists(path):
                    valid_paths.append(path)
            
            if valid_paths:
                buttons_html = "<div style='margin-top: 10px;'><strong>Open Documents:</strong><br>"
                for i, path in enumerate(valid_paths):
                    file_name = os.path.basename(path)
                    # Use a button that will trigger the file_selector
                    buttons_html += f"<button onclick='document.getElementById(\"file-download-{i}\").click()' style='margin: 5px; padding: 5px 10px; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px;'>{file_name}</button>"
                    # Add a hidden link for each file
                    buttons_html += f"<a id='file-download-{i}' href='/file={path}' download='{file_name}' style='display:none;'></a>"
                buttons_html += "</div>"
        
        return buttons_html, valid_paths
    
    # Update the respond function to use the new button creation
    def respond(message, chat_history, doc_paths_state):
        bot_message, paths = process_question(message, chat_history, doc_paths_state)
        chat_history.append((message, bot_message))
        
        # Create buttons for document access
        buttons_html, valid_paths = create_file_buttons(paths)
        
        # Update the document paths state
        doc_paths_state = valid_paths
        
        return "", chat_history, buttons_html, doc_paths_state
    
    # Set up event handlers for chat
    msg.submit(respond, [msg, chatbot, doc_paths], [msg, chatbot, doc_buttons, doc_paths])
    submit.click(respond, [msg, chatbot, doc_paths], [msg, chatbot, doc_buttons, doc_paths], queue=False)

# Launch the app
if __name__ == "__main__":
    demo.launch()
