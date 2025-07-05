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

def query_project_store(project_name, query, k_initial=7):
    """Performs an initial similarity search on a specific project's vector store."""
    if project_name not in vector_stores:
        return [] # Return empty list if project not found
    
    store = vector_stores[project_name]
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = store['index'].search(np.array(query_embedding, dtype='float32'), k_initial)
    
    initial_candidates = []
    if indices.size > 0:
        for i in indices[0]:
            if i != -1 and i < len(store['metadata']):
                initial_candidates.append(store['metadata'][i])
    return initial_candidates

def rerank_documents(query, documents, top_k=3):
    """Re-ranks a list of documents against a query using an LLM."""
    if not documents:
        return []

    # Prepare document representations for the prompt
    doc_representations = []
    for i, doc in enumerate(documents):
        # Using pdf_filename and a snippet of the final_summary for brevity in the prompt
        summary_snippet = doc.get('final_summary', '')[:200] # First 200 chars
        doc_representations.append(f"Document {i+1} (PDF: {doc.get('pdf_filename', 'Unknown')}):\nSummary Snippet: {summary_snippet}...")
    
    docs_text = "\n\n".join(doc_representations)

    prompt = f"""Given the user's query and the following list of documents, identify the top {top_k} most relevant documents to answer the query. 
Respond with a comma-separated list of document numbers (e.g., '1, 3, 5'). Only provide the numbers. If none are relevant, respond with an empty string or 'None'.

User Query: "{query}"

Documents:
{docs_text}

Top {top_k} Document Numbers:"""
    print(f"\n--- RERANKER PROMPT ---\n{prompt}\n-----------------------\n") # For debugging
    response = run_ollama(PROJECT_DETECTION_MODEL, prompt) # Using the faster model for re-ranking
    print(f"Reranker LLM Raw Response: '{response}'") # For debugging

    try:
        if not response or response.lower() == 'none':
            return []
        
        # Use regex to find numbers, accommodating 'Document X' format
        numbers_found = re.findall(r'\d+', response)
        selected_indices = [int(num) - 1 for num in numbers_found]
        
        reranked_docs = [documents[i] for i in selected_indices if 0 <= i < len(documents)]
        # Ensure unique documents if LLM mistakenly lists duplicates, and respect top_k
        final_reranked_docs = []
        seen_indices = set()
        for doc_idx in selected_indices:
            if 0 <= doc_idx < len(documents) and doc_idx not in seen_indices:
                final_reranked_docs.append(documents[doc_idx])
                seen_indices.add(doc_idx)
            if len(final_reranked_docs) >= top_k:
                break
        return final_reranked_docs
    except Exception as e:
        print(f"Error parsing reranker response: {e}. Response was: '{response}'. Falling back to original top_k.")
        return documents[:top_k]

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

    # 2. Retrieve initial candidate documents
    print("Retrieving initial documents...")
    initial_candidates = query_project_store(detected_project, user_query, k_initial=7)
    if not initial_candidates:
        return f"I couldn't find any initial documents in the {detected_project} project for your query.", detected_project, "No context used."
    print(f"  Retrieved {len(initial_candidates)} initial candidates.")
    if initial_candidates:
        print("  Initial candidate filenames:")
        for cand_doc in initial_candidates:
            print(f"    - {cand_doc.get('pdf_filename', 'Unknown PDF Filename')}")

    # 3. Re-rank documents
    print("Re-ranking documents...")
    reranked_docs = rerank_documents(user_query, initial_candidates, top_k=3)
    if not reranked_docs:
        return f"Could not find sufficiently relevant documents in {detected_project} after re-ranking.", detected_project, "No context used after re-ranking."
    print(f"  Re-ranked to {len(reranked_docs)} documents.")

    # 4. Prepare context from re-ranked documents
    context_list = []
    for doc in reranked_docs:
        context_list.append(f"PDF: {doc['pdf_filename']}\nFinal Summary: {doc['final_summary']}\nPage Summaries: {doc['page_summaries']}")
    
    context_for_llm = "\n\n---\n\n".join(context_list)
    context_for_ui = "\n\n".join([f"- {doc['pdf_filename']}" for doc in reranked_docs])

    # 5. Generate the answer
    print("Generating answer...")
    final_prompt = f"""You are a helpful assistant. Answer the user's question based *only* on the context provided below. If the answer is not in the context, say so. Be concise and clear.

Context:
{context_for_llm}

User Question: {user_query}

Answer:"""
    
    final_answer = run_ollama(ANSWER_GENERATION_MODEL, final_prompt)
    print("  Answer generated.")
    
    return final_answer, detected_project, f"Used {len(reranked_docs)} documents:\n{context_for_ui}"


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
