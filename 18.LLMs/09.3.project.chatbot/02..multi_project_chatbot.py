# This script requires:
# pip install gradio sentence-transformers faiss-cpu numpy

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import gradio as gr
import subprocess
import re

# === CONFIGURATION ===
VECTOR_STORE_DIR = r'D:\OEssam\for.3.project.chatbot'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
PROJECT_DETECTION_MODEL = 'phi3:mini' # A small, fast model for classification
ANSWER_GENERATION_MODEL = 'gemma3:latest' # A powerful model for generation

# === GLOBAL VARIABLES ===
vector_stores = {}
embedder = None

# === CORE FUNCTIONS ===

def load_vector_stores():
    """Loads all FAISS indexes and metadata from the specified directory."""
    global embedder, vector_stores
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Loading vector stores from: {VECTOR_STORE_DIR}")
    files = os.listdir(VECTOR_STORE_DIR)
    faiss_files = [f for f in files if f.endswith('.faiss')]
    
    for faiss_file in faiss_files:
        base_name = faiss_file.replace('.faiss', '')
        # Recreate the original project name from the safe filename
        project_name = base_name.replace('_', ' ').title()
        # Handle specific cases like 'Eipico 3'
        project_name = re.sub(r'(\d+)', r' \1', project_name).replace('Eipico', 'Eipico')

        index_path = os.path.join(VECTOR_STORE_DIR, faiss_file)
        metadata_path = os.path.join(VECTOR_STORE_DIR, f"{base_name}_metadata.pkl")
        
        if os.path.exists(metadata_path):
            print(f"  Loading store for project: {project_name}")
            index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f_meta:
                metadata = pickle.load(f_meta)
            vector_stores[project_name] = {'index': index, 'metadata': metadata}
        else:
            print(f"  [WARNING] Metadata for {faiss_file} not found. Skipping.")
    print(f"Loaded {len(vector_stores)} vector stores.")

def run_ollama(model, prompt):
    """Runs a prompt through a specified Ollama model."""
    process = subprocess.Popen([
        "ollama", "run", model
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    stdout, stderr = process.communicate(prompt)
    if stderr:
        print(f"Ollama Error ({model}): {stderr}")
    return stdout.strip()

def detect_project(query):
    """Uses an LLM to detect which project the query is about."""
    project_names = list(vector_stores.keys())
    prompt = f"""Your task is to identify which of the following projects a user's question is about. Respond with ONLY the project name from the list and nothing else. If the question is ambiguous or not about any project, respond with 'Unknown'.

Projects: {', '.join(project_names)}

User Question: "{query}"

Project Name:"""
    
    detected_name = run_ollama(PROJECT_DETECTION_MODEL, prompt)
    
    # Find the best match in our list of project names
    for name in project_names:
        if name.lower() in detected_name.lower():
            return name
    return "Unknown"

def query_project_store(project_name, query, k=3):
    """Performs a similarity search on a specific project's vector store."""
    if project_name not in vector_stores:
        return [], []
    
    store = vector_stores[project_name]
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = store['index'].search(np.array(query_embedding, dtype='float32'), k)
    
    results = []
    context_docs = []
    if indices.size > 0:
        for i in indices[0]:
            if i != -1:
                doc = store['metadata'][i]
                results.append(doc)
                context_docs.append(f"PDF: {doc['pdf_filename']}\nFinal Summary: {doc['final_summary']}\nPage Summaries: {doc['page_summaries']}")
    return results, context_docs

# === GRADIO INTERFACE ===

def chat_interface(user_query):
    if not user_query:
        return "Please ask a question.", "", ""

    print(f"Received query: {user_query}")
    
    # 1. Detect the project
    print("Detecting project...")
    detected_project = detect_project(user_query)
    if detected_project == "Unknown":
        return "Could not determine the project from your question. Please be more specific (e.g., 'In the R5 project, ...').", "Unknown", "No context used."
    print(f"  Project detected: {detected_project}")

    # 2. Retrieve relevant documents
    print("Retrieving documents...")
    retrieved_docs, context_list = query_project_store(detected_project, user_query, k=3)
    if not retrieved_docs:
        return f"I couldn't find any relevant documents in the {detected_project} project for your query.", detected_project, "No context used."
    
    context_for_llm = "\n\n---\n\n".join(context_list)
    context_for_ui = "\n\n".join([f"- {doc['pdf_filename']}" for doc in retrieved_docs])
    print(f"  Retrieved {len(retrieved_docs)} documents.")

    # 3. Generate the answer
    print("Generating answer...")
    final_prompt = f"""You are a helpful assistant. Answer the user's question based *only* on the context provided below. If the answer is not in the context, say so. Be concise and clear.

Context:
{context_for_llm}

User Question: {user_query}

Answer:"""
    
    final_answer = run_ollama(ANSWER_GENERATION_MODEL, final_prompt)
    print("  Answer generated.")
    
    return final_answer, detected_project, f"Used {len(retrieved_docs)} documents:\n{context_for_ui}"


if __name__ == "__main__":
    # Load data once on startup
    load_vector_stores()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Multi-Project RAG Chatbot")
        gr.Markdown("Ask a question about the Damietta Buildings, Eipico 3, or R5 projects.")
        
        with gr.Row():
            with gr.Column(scale=2):
                question_box = gr.Textbox(label="Your Question", placeholder="e.g., What are the key terms in the El-Barakah contract in the R5 project?")
                submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                detected_project_box = gr.Textbox(label="Detected Project", interactive=False)
                context_box = gr.Textbox(label="Context Documents Used", interactive=False, lines=5)

        answer_box = gr.Markdown(label="Answer")

        submit_btn.click(
            fn=chat_interface,
            inputs=[question_box],
            outputs=[answer_box, detected_project_box, context_box]
        )

    print("Launching Gradio interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
