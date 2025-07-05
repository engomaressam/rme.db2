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

PROMPT_TEMPLATE = '''
Given the following purchase order or contract text, extract the following fields as a JSON object:
- payment_terms
- delivery_terms
- special_conditions
If a field is not found, return it as an empty string.

Text: """
{doc}
"""

Return ONLY a JSON object with these keys in English.
'''

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
    print("Prompt sent to LLM:\n----------------------")
    prompt = PROMPT_TEMPLATE.format(doc=row['extracted_text'])
    print(prompt)
    print("\n--- SENDING TO LLM ---\n")
    llm_response = query_ollama(prompt)
    print("LLM RAW RESPONSE:\n------------------")
    print(llm_response)

if __name__ == '__main__':
    main()
