import pymysql
import requests
import json
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# MySQL connection settings
HOST = '10.10.11.242'
USER = 'omar2'
PASSWORD = 'Omar_54321'
DB = 'RME_TEST'
TABLE = 'po.pdfs'

# Ollama API settings
OLLAMA_URL = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'dolphin-mixtral'

SINGLE_FIELD_PROMPTS = {
    'payment_terms': 'Extract the payment terms from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""',
    'delivery_terms': 'Extract the delivery terms from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""',
    'special_conditions': 'Extract any special conditions from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""',
}

def get_sample_row():
    conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DB, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT id, extracted_text FROM `{TABLE}` WHERE extracted_text IS NOT NULL AND LENGTH(extracted_text) > 20 LIMIT 1")
        row = cursor.fetchone()
    conn.close()
    return row

def query_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get('response', '').strip()
    except Exception as e:
        return f"[ERROR: {e}]"

def main():
    row = get_sample_row()
    if not row:
        print("No sample row found with extracted text.")
        return
    for field, prompt_template in SINGLE_FIELD_PROMPTS.items():
        prompt = prompt_template.format(doc=row['extracted_text'])
        print(f"\nPrompt for {field}:\n{'-'*40}\n{prompt}\n--- SENDING TO LLM ---")
        llm_response = query_ollama(prompt)
        print(f"LLM response for {field}:\n{'-'*40}\n{llm_response}\n")

if __name__ == '__main__':
    main()
