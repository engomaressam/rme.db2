import os
import json
import glob
import pandas as pd
from tabulate import tabulate

# Paths
EXTRACTED_DIR = r'D:/OEssam/extracted_json'
CHROMA_DB_DIR = r'D:/OEssam/chroma_db'
TUNING_DIR = r'C:/Users/Omar Essam2/OneDrive - Rowad Modern Engineering/x004 Data Science/03.rme.db/05.llm/gpt.tuning'
TUNING_FILE = os.path.join(TUNING_DIR, 'gpt_feedback_log.jsonl')

print("\n=== Extracted JSONs ===")
if os.path.exists(EXTRACTED_DIR):
    json_files = glob.glob(os.path.join(EXTRACTED_DIR, '*.json'))
    print(f"Found {len(json_files)} extracted JSON files:")
    for f in json_files:
        print(f"- {os.path.basename(f)} (size: {os.path.getsize(f)//1024} KB)")
else:
    print("[WARN] Extracted JSON directory not found.")

print("\n=== Chroma DB ===")
if os.path.exists(CHROMA_DB_DIR):
    files = os.listdir(CHROMA_DB_DIR)
    print(f"Chroma DB folder contains {len(files)} files/folders:")
    for f in files:
        print(f"- {f}")
else:
    print("[WARN] Chroma DB directory not found.")

print("\n=== GPT Training Data ===")
if os.path.exists(TUNING_FILE):
    with open(TUNING_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"Found {len(lines)} Q&A pairs in training data.")
    # Show as table
    qa_list = [json.loads(line) for line in lines]
    df = pd.DataFrame(qa_list)
    if not df.empty:
        print("\nSample Q&A Table:")
        print(tabulate(df[['question', 'answer']], headers='keys', tablefmt='psql', showindex=True, maxcolwidths=[40, 60]))
    else:
        print("[INFO] No Q&A data to display.")
else:
    print("[WARN] GPT training data file not found.") 