import os
import glob
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess
import gradio as gr

# === CONFIG ===
WORD_DOCS_DIR = r'D:\OEssam\Test\gemma3-3rd-time'
DEFAULT_MODEL = 'phi3:latest'  # You can change to 'llama3:latest' or any other Ollama model
EMBED_MODEL = 'all-MiniLM-L6-v2'
TOP_K = 3

# === 1. Extract summaries from Word files ===
def extract_summaries(word_docs_dir):
    summaries = []
    for path in glob.glob(os.path.join(word_docs_dir, '*.docx')):
        doc = Document(path)
        summary_text = ''
        in_summary = False
        for para in doc.paragraphs:
            if 'Final Comprehensive Summary' in para.text:
                in_summary = True
                continue
            if in_summary:
                if para.text.strip() == '':
                    break
                summary_text += para.text + '\n'
        if summary_text.strip():
            summaries.append({'doc': os.path.basename(path), 'path': path, 'summary': summary_text.strip()})
    return summaries

# === 2. Embed summaries ===
def embed_texts(texts, model):
    return model.encode(texts, convert_to_tensor=True).cpu().numpy()

# === 3. Query LLM ===
def query_llm(context, question, model_tag=DEFAULT_MODEL):
    prompt = f"You are an assistant for the Damietta Buildings Project. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    process = subprocess.Popen([
        "ollama", "run", model_tag
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(prompt.encode("utf-8"))
    out = stdout.decode("utf-8", errors="ignore")
    err = stderr.decode("utf-8", errors="ignore")
    if err:
        print(f"[LLM ERROR]: {err}")
    return out.strip()

# === 4. Gradio Chatbot Interface ===
class DamiettaChatbot:
    def __init__(self, word_docs_dir, embed_model_name, default_model_tag):
        self.summaries = extract_summaries(word_docs_dir)
        self.texts = [s['summary'] for s in self.summaries]
        self.embedder = SentenceTransformer(embed_model_name)
        self.summary_embeds = embed_texts(self.texts, self.embedder)
        self.model_tag = default_model_tag

    def ask(self, question, model_tag=None):
        if not question.strip():
            return "Please enter a question."
        if not self.summaries:
            return "No summaries found. Please check your Word docs directory."
        if model_tag is None:
            model_tag = self.model_tag
        q_embed = embed_texts([question], self.embedder)
        sims = cosine_similarity(q_embed, self.summary_embeds)[0]
        top_idx = np.argsort(sims)[::-1][:TOP_K]
        context = '\n\n'.join([self.summaries[i]['summary'] for i in top_idx])
        answer = query_llm(context, question, model_tag)
        return answer

chatbot = DamiettaChatbot(WORD_DOCS_DIR, EMBED_MODEL, DEFAULT_MODEL)

sample_questions = [
    "What is the contract value for the main project?",
    "Who are the parties involved in the Damietta Buildings Project?",
    "List key obligations mentioned in the contracts.",
    "Summarize the most important terms for the Abo Taleb Company contract.",
    "Are there any deadlines or important dates in the project documents?"
]

def gradio_chat_interface(question, model):
    return chatbot.ask(question, model)

demo = gr.Interface(
    fn=gradio_chat_interface,
    inputs=[
        gr.Textbox(lines=2, label="Ask about Damietta Buildings Project"),
        gr.Dropdown(choices=["phi3:latest", "llama3:latest", "gemma3:latest"], value="phi3:latest", label="Model (Ollama)")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Damietta Buildings Project Chatbot",
    description="Ask questions about the Damietta Buildings Project contracts. Uses retrieval-augmented generation (RAG) over extracted summaries.",
    examples=[[q, "phi3:latest"] for q in sample_questions],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)
