import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import requests

app = Flask(__name__)

# Configuration
OLLAMA_API_URL = "http://10.10.12.202:11434/api/generate"  # GPU machine's IP
MODEL_NAME = "mistral"  # Using mistral model which is available on the server

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
    # Get all PDF files from the Alstom directory
    base_paths = ["C:\\alstom\\folder1", "C:\\alstom\\folder2"]
    pdf_files = []
    
    for base_path in base_paths:
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    simplified_path = os.path.relpath(os.path.dirname(full_path), base_path)
                    pdf_files.append({
                        'name': file,
                        'full_path': full_path,
                        'simplified_path': simplified_path
                    })
    
    return pdf_files

# Generate answer using Ollama
def generate_answer(question, context):
    prompt = f"""You are an expert assistant for the Alstom project. Answer the following question based on your knowledge of the project documents. Be concise and accurate.

Context about the project documents:
{context}

Question: {question}

Answer:"""
    
    try:
        # Try the generate API
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        
        return "I'm having trouble connecting to the language model. Please try again later."
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return f"Error: {str(e)}"

# Global variables to store data
summaries = {}
documents = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    global summaries, documents
    
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'answer': 'Please ask a question.', 'sources': []})
    
    # Simple keyword matching to find relevant documents
    question_keywords = question.lower().split()
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
    for doc in top_docs:
        context += f"\nDocument: {doc['name']} (in {doc['simplified_path']})\n"
        
        # Add summary if available
        if doc['name'] in summaries:
            context += f"Summary: {summaries[doc['name']]}\n\n"
    
    # Generate answer
    answer = generate_answer(question, context)
    
    # Return the answer and sources
    sources = [doc['name'] for doc in top_docs]
    return jsonify({'answer': answer, 'sources': sources})

if __name__ == '__main__':
    # Load data on startup
    print("Loading document summaries...")
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} document summaries")
    
    print("Loading document information...")
    documents = load_document_info()
    print(f"Loaded information for {len(documents)} documents")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
