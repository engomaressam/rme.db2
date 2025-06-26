import os
import glob
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess

# === CONFIG ===
WORD_DOCS_DIR = r'D:\OEssam\Test\gemma3-3rd-time'
LLM_MODEL_TAG = 'gemma3:latest'
EMBED_MODEL = 'all-MiniLM-L6-v2'  # You can change to a local model if desired
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
    return model.encode(texts, convert_to_tensor=True)

# === 3. Query LLM ===
def query_llm(context, question):
    prompt = f"You are an assistant for the Damietta Buildings Project. Use the following context to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    process = subprocess.Popen([
        "ollama", "run", LLM_MODEL_TAG
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(prompt.encode("utf-8"))
    out = stdout.decode("utf-8", errors="ignore")
    err = stderr.decode("utf-8", errors="ignore")
    if err:
        print(f"[LLM ERROR]: {err}")
    return out.strip()

# === 4. Chatbot loop ===
def main():
    print("Loading summaries...")
    summaries = extract_summaries(WORD_DOCS_DIR)
    if not summaries:
        print("No summaries found. Please check your Word docs directory.")
        return
    print(f"Loaded {len(summaries)} summaries.")
    texts = [s['summary'] for s in summaries]
    print("Embedding summaries...")
    embedder = SentenceTransformer(EMBED_MODEL)
    summary_embeds = embed_texts(texts, embedder)
    print("Ready! Type your questions. Type 'exit' to quit.")
    while True:
        question = input("\nAsk about Damietta Buildings Project: ").strip()
        if question.lower() in {'exit', 'quit'}:
            break
        q_embed = embed_texts([question], embedder)
        sims = cosine_similarity(q_embed, summary_embeds)[0]
        top_idx = np.argsort(sims)[::-1][:TOP_K]
        context = '\n\n'.join([summaries[i]['summary'] for i in top_idx])
        answer = query_llm(context, question)
        print(f"\n[Gemma3]: {answer}\n")

if __name__ == "__main__":
    main()
