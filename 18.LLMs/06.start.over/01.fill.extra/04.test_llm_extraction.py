import pymysql
import requests
import random
import json
import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter

# Set UTF-8 encoding for stdout if possible (Python 3.7+)
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

# Output directory
OUTPUT_DIR = r'D:\OEssam\Test'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_random_rows(conn, n=5):
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT id FROM `{TABLE}` WHERE extracted_text IS NOT NULL AND LENGTH(extracted_text) > 20 AND pdf_path IS NOT NULL AND pdf_path != ''")
        all_ids = [row['id'] for row in cursor.fetchall()]
        sample_ids = random.sample(all_ids, min(n, len(all_ids)))
        format_strings = ','.join(['%s'] * len(sample_ids))
        cursor.execute(f"SELECT id, extracted_text, pdf_path, pdf_filename FROM `{TABLE}` WHERE id IN ({format_strings})", tuple(sample_ids))
        return cursor.fetchall()

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

import textwrap

def create_cover_pdf(title, answer, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, title)
    c.setFont("Helvetica", 12)
    y = height - 90
    wrap_width = 90
    for line in answer.splitlines():
        wrapped = textwrap.wrap(line, width=wrap_width)
        for subline in wrapped:
            c.drawString(50, y, subline)
            y -= 20
            if y < 50:
                c.showPage()
                y = height - 50
    c.save()

def merge_pdfs(cover_pdfs, original_pdf, output_pdf):
    writer = PdfWriter()
    # Add all cover pages
    for cover_pdf in cover_pdfs:
        cover_reader = PdfReader(cover_pdf)
        for page in cover_reader.pages:
            writer.add_page(page)
    # Add original PDF pages
    orig_reader = PdfReader(original_pdf)
    for page in orig_reader.pages:
        writer.add_page(page)
    with open(output_pdf, 'wb') as f:
        writer.write(f)

def main():
    print("Connecting to MySQL...")
    conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DB, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    print("Selecting 100 random rows with extracted text and pdf_path...")
    rows = get_random_rows(conn, 100)
    print(f"Processing {len(rows)} rows...")
    debug_log = []
    QUESTIONS = [
        ("Payment Terms", 'Extract the payment terms from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
        ("Delivery Terms", 'Extract the delivery terms from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
        ("Special Conditions", 'Extract any special conditions from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
    ]
    for i, row in enumerate(rows, 1):
        text = row['extracted_text']
        pdf_path = row['pdf_path']
        pdf_id = row['id']
        pdf_filename = row['pdf_filename']
        try:
            preview = f"\nRow {i} (ID: {pdf_id}):\nText preview: {text[:100].replace(chr(10), ' ')}...\nPDF Path: {pdf_path}"
            print(preview.encode('utf-8', errors='replace').decode('cp1252', errors='replace'))
        except Exception:
            print(f"Row {i} (ID: {pdf_id}): (unprintable preview) PDF Path: {pdf_path}")
        cover_pdfs = []
        answers = {}
        for title, prompt_template in QUESTIONS:
            prompt = prompt_template.format(doc=text)
            print(f"\nPrompt for {title}:\n{'-'*40}\n{prompt}\n--- SENDING TO LLM ---")
            llm_result = query_ollama(prompt)
            print(f"LLM response for {title}:\n{'-'*40}\n{llm_result}\n")
            answers[title] = llm_result
            # Create cover page for this answer
            cover_pdf_path = os.path.join(OUTPUT_DIR, f"cover_{pdf_id}_{title.replace(' ', '_')}.pdf")
            create_cover_pdf(f"Mixtral: {title}", llm_result, cover_pdf_path)
            cover_pdfs.append(cover_pdf_path)
        # Merge all cover pages and original PDF
        output_pdf_path = os.path.join(OUTPUT_DIR, f"po_{pdf_id}.pdf")
        if os.path.exists(pdf_path):
            try:
                merge_pdfs(cover_pdfs, pdf_path, output_pdf_path)
                print(f"Combined PDF saved: {output_pdf_path}")
            except Exception as e:
                print(f"[ERROR merging PDFs for ID {pdf_id}]: {e}")
        else:
            print(f"[ERROR: PDF not found at {pdf_path}]")
        # Clean up covers
        for cover_pdf_path in cover_pdfs:
            if os.path.exists(cover_pdf_path):
                os.remove(cover_pdf_path)
        debug_log.append({'id': pdf_id, 'answers': answers})
    # Save debug log
    with open(os.path.join(OUTPUT_DIR, 'llm_debug_log.json'), 'w', encoding='utf-8') as f:
        json.dump(debug_log, f, ensure_ascii=False, indent=2)

    conn.close()

if __name__ == '__main__':
    main()
