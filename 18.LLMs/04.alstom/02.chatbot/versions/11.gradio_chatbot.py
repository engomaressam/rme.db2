import os
import json
import gradio as gr
import requests
import time

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

# Load data
print("Loading document summaries...")
summaries = load_summaries()
print(f"Loaded {len(summaries)} document summaries")

print("Loading document information...")
documents = load_document_info()
print(f"Loaded information for {len(documents)} documents")

# Function to process a question and return an answer
def process_question(message, history):
    # Simple keyword matching to find relevant documents
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
    
    for doc in top_docs:
        context += f"\nDocument: {doc['name']} (in {doc['simplified_path']})\n"
        sources.append(f"{doc['name']} (in {doc['simplified_path']})")
        
        # Add summary if available
        if doc['name'] in summaries:
            context += f"Summary: {summaries[doc['name']]}\n\n"
    
    # Generate answer
    answer = generate_answer(message, context)
    
    # Add sources to the answer
    if sources:
        answer += "\n\nSources:\n" + "\n".join([f"- {source}" for source in sources])
    
    return answer

# Create the Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # Alstom Project Assistant
        
        Ask questions about the Alstom project documents and get answers based on the project documentation.
        """
    )
    
    # Add logo
    gr.Image("static/logo.png", width=200, show_label=False)
    
    chatbot = gr.Chatbot(
        show_label=False,
        bubble_full_width=False,
        height=500,
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question about the Alstom project...",
            show_label=False,
            scale=9,
        )
        submit = gr.Button("Send", scale=1)
    
    # Set up event handlers
    msg.submit(process_question, [msg, chatbot], [chatbot, msg])
    submit.click(process_question, [msg, chatbot], [chatbot, msg], queue=False)
    
    # Clear the textbox after sending
    msg.submit(lambda: "", None, [msg])
    submit.click(lambda: "", None, [msg])
    
    # Add a welcome message
    gr.on_load(lambda: chatbot.append(["", "Welcome to the Alstom Project Assistant! I can answer questions about the project documents. How can I help you today?"]))

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
