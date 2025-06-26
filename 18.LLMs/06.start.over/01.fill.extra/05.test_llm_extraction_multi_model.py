import pymysql
import requests
import random
import json
import os
import sys
import textwrap
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PyPDF2 import PdfReader, PdfWriter
import time
import arabic_reshaper
from bidi.algorithm import get_display

# Register a Unicode font for Arabic and English
ARABIC_FONT_PATH = r'C:\Windows\Fonts\Amiri-Regular.ttf'
ARABIC_FONT_NAME = 'Amiri'
pdfmetrics.registerFont(TTFont(ARABIC_FONT_NAME, ARABIC_FONT_PATH))

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
OLLAMA_MODELS = [
    ('Mixtral', 'dolphin-mixtral'),
    ('ALLaM', 'iKhalid/ALLaM:7b'),
    ('Command-r7b-arabic', 'command-r7b-arabic')
]

QUESTIONS = [
    ("Payment Terms", 'Extract the payment terms from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
    ("Delivery Terms", 'Extract the delivery terms from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
    ("Special Conditions", 'Extract any special conditions from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
]

OUTPUT_DIR = r'D:\OEssam\Test'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- LayoutLM OCR Extraction ---
from pdf2image import convert_from_path
from pytesseract import image_to_string, pytesseract
pytesseract.tesseract_cmd = r'C:\Users\Omar Essam2\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

# --- EasyOCR Extraction ---
import easyocr
EASY_OCR_READER = easyocr.Reader(['ar', 'en'], gpu=False)

def layoutlm_ocr(pdf_path):
    poppler_path = r'C:\Program Files\poppler\Library\bin'
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
    except Exception as e:
        print(f"[ERROR] Could not convert PDF to images: {e}")
        return ''
    all_text = []
    for i, img in enumerate(images):
        text = image_to_string(img, lang='ara+eng')
        all_text.append(text)
    return '\n'.join(all_text)

def easyocr_extract(pdf_path):
    poppler_path = r'C:\Program Files\poppler\Library\bin'
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
    except Exception as e:
        print(f"[ERROR] Could not convert PDF to images: {e}")
        return ''
    all_text = []
    for i, img in enumerate(images):
        # Convert PIL image to numpy array for EasyOCR
        result = EASY_OCR_READER.readtext(np.array(img))
        page_text = '\n'.join([x[1] for x in result])
        all_text.append(page_text)
    return '\n'.join(all_text)

def get_random_rows(conn, n=5):
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT id, pdf_path, extracted_text, pdf_filename FROM `{TABLE}` WHERE extracted_text IS NOT NULL AND LENGTH(extracted_text) > 20 AND pdf_path IS NOT NULL AND pdf_path != '' AND document_type = 'Purchase Order'")
        all_rows = cursor.fetchall()
        if not all_rows:
            return []
        sample_rows = random.sample(all_rows, min(n, len(all_rows)))
        return sample_rows
        format_strings = ','.join(['%s'] * len(sample_ids))
        cursor.execute(f"SELECT id, extracted_text, pdf_path, pdf_filename FROM `{TABLE}` WHERE id IN ({format_strings})", tuple(sample_ids))
        return cursor.fetchall()

def query_ollama(model, prompt, max_retries=3, retry_delay=5):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get('response', '').strip()
        except Exception as e:
            if attempt < max_retries:
                print(f"[WARN] LLM call failed (attempt {attempt}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                return f"[ERROR: {e}]"

def shape_arabic_text(text):
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text

def create_cover_pdf(model, extraction_method, question, answer, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    c.setFont(ARABIC_FONT_NAME, 14)
    # Label at the top: Model | Extraction Method
    label = f"Model: {model} | Extraction: {extraction_method}"
    c.drawString(50, height - 40, shape_arabic_text(label))
    # Question title
    c.setFont(ARABIC_FONT_NAME, 13)
    c.drawString(50, height - 70, shape_arabic_text(question))
    c.setFont(ARABIC_FONT_NAME, 12)
    y = height - 110
    wrap_width = 90
    for line in answer.splitlines():
        shaped_line = shape_arabic_text(line)
        wrapped = textwrap.wrap(shaped_line, width=wrap_width)
        for subline in wrapped:
            c.drawString(50, y, subline)
            y -= 20
            if y < 50:
                c.showPage()
                y = height - 50
    c.save()

def create_text_pdf(title, text, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    c.setFont(ARABIC_FONT_NAME, 16)
    shaped_title = shape_arabic_text(title)
    c.drawString(50, height - 50, shaped_title)
    c.setFont(ARABIC_FONT_NAME, 12)
    y = height - 90
    wrap_width = 90
    for line in text.splitlines():
        shaped_line = shape_arabic_text(line)
        wrapped = textwrap.wrap(shaped_line, width=wrap_width)
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
    print("Selecting 5 random rows with extracted text and pdf_path...")
    rows = get_random_rows(conn, 5)
    print(f"Processing {len(rows)} rows...")
    debug_log = []
    for i, row in enumerate(rows, 1):
        pdf_path = row['pdf_path']
        pdf_id = row['id']
        pdf_filename = row['pdf_filename']
        extraction_methods = [
            ("Original OCR", row['extracted_text']),
            ("LayoutLM OCR", layoutlm_ocr(pdf_path) if os.path.exists(pdf_path) else ""),
            ("EasyOCR", easyocr_extract(pdf_path) if os.path.exists(pdf_path) else "")
        ]
        cover_pdfs = []
        answers = {}
        for extraction_method, extracted_text in extraction_methods:
            # Defensive: decode bytes if needed
            if isinstance(extracted_text, bytes):
                try:
                    extracted_text = extracted_text.decode('utf-8')
                except Exception:
                    pass
            # 1. LLM answers (9 pages)
            for model_name, model_id in OLLAMA_MODELS:
                for q_title, prompt_template in QUESTIONS:
                    prompt = prompt_template.format(doc=extracted_text)
                    print(f"\n[DIAG] {extraction_method} | Prompt for {model_name} - {q_title} type: {type(prompt)}")
                    print(f"[DIAG] Prompt content (first 200 chars): {repr(prompt[:200])}")
                    llm_result = query_ollama(model_id, prompt)
                    print(f"[DIAG] LLM response type: {type(llm_result)}")
                    print(f"[DIAG] LLM response (first 200 chars): {repr(llm_result[:200])}")
                    answers[f"{extraction_method}_{model_name}_{q_title}"] = llm_result
                    cover_pdf_path = os.path.join(OUTPUT_DIR, f"cover_{pdf_id}_{extraction_method.replace(' ','_')}_{model_name}_{q_title.replace(' ', '_')}.pdf")
                    create_cover_pdf(model_name, extraction_method, q_title, llm_result, cover_pdf_path)
                    cover_pdfs.append(cover_pdf_path)
            # 2. Add extracted text page for this method
            text_pdf_path = os.path.join(OUTPUT_DIR, f"text_{pdf_id}_{extraction_method.replace(' ','_')}.pdf")
            create_text_pdf(f"Extracted Text ({extraction_method})", extracted_text, text_pdf_path)
            cover_pdfs.append(text_pdf_path)
        # 3. Merge all covers + original
        output_pdf_path = os.path.join(OUTPUT_DIR, f"po_{pdf_id}_all_models.pdf")
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
    with open(os.path.join(OUTPUT_DIR, 'llm_debug_log_multi_model.json'), 'w', encoding='utf-8') as f:
        json.dump(debug_log, f, ensure_ascii=False, indent=2)
    conn.close()

if __name__ == '__main__':
    main()
