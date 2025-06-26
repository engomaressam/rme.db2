import os
import json
import requests
import numpy as np
from typing import List, Dict, Any
import PyPDF2
from sentence_transformers import SentenceTransformer
import re
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
OLLAMA_API_URL = "http://10.10.12.202:11434/api/generate"  # GPU machine's IP
MODEL_NAME = "mistral"  # Using mistral model which is available on the server
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight embedding model
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
TOP_K_CHUNKS = 5  # Number of chunks to retrieve for each query

class AlstomChatbot:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.embedding_model = None
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.index_path = os.path.join(self.data_dir, "document_index.pkl")
        
    def initialize(self, force_rebuild=False):
        """Initialize the chatbot by loading or building the document index"""
        print("Initializing Alstom Chatbot...")
        
        # Load or create embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Check if index exists
        if os.path.exists(self.index_path) and not force_rebuild:
            print("Loading existing document index...")
            self.load_index()
        else:
            print("Building new document index...")
            self.build_index()
    
    def load_index(self):
        """Load the document index from disk"""
        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.chunks = data['chunks']
                self.embeddings = data['embeddings']
            print(f"Loaded index with {len(self.documents)} documents and {len(self.chunks)} chunks")
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            print("Building new index instead...")
            self.build_index()
    
    def build_index(self):
        """Build a new document index"""
        # Get all PDF files
        pdf_files = self.get_all_pdf_files()
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        self.documents = []
        self.chunks = []
        
        for i, pdf_info in enumerate(pdf_files, 1):
            print(f"Processing {i}/{len(pdf_files)}: {pdf_info['name']}")
            
            # Extract text
            text = self.extract_pdf_text(pdf_info['full_path'])
            if not text:
                continue
                
            # Add to documents
            doc_id = len(self.documents)
            self.documents.append({
                'id': doc_id,
                'name': pdf_info['name'],
                'path': pdf_info['full_path'],
                'simplified_path': pdf_info['simplified_path']
            })
            
            # Chunk the text
            doc_chunks = self.chunk_text(text, doc_id)
            self.chunks.extend(doc_chunks)
            
            # Save progress periodically
            if i % 10 == 0 or i == len(pdf_files):
                self.save_index()
        
        # Create embeddings for all chunks
        print(f"Creating embeddings for {len(self.chunks)} chunks...")
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        self.embeddings = self.embedding_model.encode(chunk_texts)
        
        # Save the final index
        self.save_index()
    
    def get_all_pdf_files(self):
        """Get all PDF files from the Alstom directory"""
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
    
    def extract_pdf_text(self, pdf_path):
        """Extract text content from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return None
    
    def chunk_text(self, text, doc_id):
        """Split text into overlapping chunks"""
        chunks = []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Create chunks with overlap
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            if len(chunk_text) < 50:  # Skip very small chunks
                continue
                
            chunks.append({
                'doc_id': doc_id,
                'text': chunk_text,
                'start': i,
                'end': i + len(chunk_text)
            })
        
        return chunks
    
    def save_index(self):
        """Save the document index to disk"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
        
        print(f"Saved index with {len(self.documents)} documents and {len(self.chunks)} chunks")
    
    def query(self, question):
        """Answer a question about the documents"""
        # Check if index is loaded
        if not self.documents or not self.chunks or not self.embeddings.any():
            print("Document index not loaded. Initializing...")
            self.initialize()
        
        # Embed the question
        question_embedding = self.embedding_model.encode([question])[0]
        
        # Find similar chunks
        similarities = cosine_similarity([question_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-TOP_K_CHUNKS:][::-1]
        
        # Get the top chunks
        top_chunks = [self.chunks[i] for i in top_indices]
        
        # Format context for the LLM
        context = ""
        doc_ids_used = set()
        
        for chunk in top_chunks:
            doc_id = chunk['doc_id']
            doc = self.documents[doc_id]
            
            if doc_id not in doc_ids_used:
                context += f"\nDocument: {doc['name']} (in {doc['simplified_path']})\n"
                doc_ids_used.add(doc_id)
            
            context += f"Content: {chunk['text'][:500]}...\n\n"
        
        # Generate answer using Ollama
        answer = self.generate_answer(question, context)
        
        # Return the answer and sources
        sources = [self.documents[chunk['doc_id']]['name'] for chunk in top_chunks]
        return answer, list(set(sources))
    
    def generate_answer(self, question, context):
        """Generate an answer using Ollama"""
        prompt = f"""You are an expert assistant for the Alstom project. Answer the following question based ONLY on the provided context. Be concise and accurate. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer that question."

Context:
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
            
            # If that fails, try the completions API
            completions_url = OLLAMA_API_URL.replace('/generate', '/completions')
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "max_tokens": 500
            }
            
            response = requests.post(completions_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('choices', [{}])[0].get('text', '')
            
            return "I'm having trouble connecting to the language model. Please try again later."
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return f"Error: {str(e)}"

def main():
    print("Alstom Project Chatbot")
    print("======================")
    print("This chatbot can answer questions about the Alstom project documents.")
    print("Type 'exit' to quit.")
    print()
    
    chatbot = AlstomChatbot()
    chatbot.initialize()
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        if not question.strip():
            continue
        
        print("\nSearching documents...")
        answer, sources = chatbot.query(question)
        
        print("\nAnswer:")
        print(answer)
        
        print("\nSources:")
        for source in sources:
            print(f"- {source}")

if __name__ == "__main__":
    main()
