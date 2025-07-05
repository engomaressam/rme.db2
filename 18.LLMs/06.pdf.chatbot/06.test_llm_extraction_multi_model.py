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
import numpy as np
import cv2
import easyocr
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import image_to_string, pytesseract

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
    ('Llama4', 'llama4'),
    ('Command-r7b-arabic', 'command-r7b-arabic')
]

QUESTIONS = [
    ("Payment Terms", 'Extract the payment terms from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
    ("Delivery Terms", 'Extract the delivery terms from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
    ("Special Conditions", 'Extract any special conditions from the following purchase order or contract text. If not found, reply with "Not found". Text: """{doc}"""'),
]

OUTPUT_DIR = r'D:\OEssam\Test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

pytesseract.tesseract_cmd = r'C:\Users\Omar Essam2\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
print("Initializing EasyOCR with GPU support...")
EASY_OCR_READER = easyocr.Reader(['ar', 'en'], gpu=True)

def preprocess_image(pil_img):
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_denoised = cv2.fastNlMeansDenoising(img_bin, h=30)
    return img_denoised

def layoutlm_ocr(pdf_path):
    poppler_path = r'C:\Program Files\poppler\Library\bin'
    try:
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
    except Exception as e:
        print(f"[ERROR] Could not convert PDF to images: {e}")
        return ''
    all_text = []
    for i, img in enumerate(images):
        pre_img = preprocess_image(img)
        pil_img = Image.fromarray(pre_img)
        text = image_to_string(pil_img, lang='ara+eng')
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
        pre_img = preprocess_image(img)
        result = EASY_OCR_READER.readtext(pre_img)
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

def shape_arabic_text(text):
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    except Exception:
        return text

# Helper to draw Arabic or Unicode text (reshapes, applies bidi, sets font)
def draw_unicode_text(canvas_obj, text, x, y, font=ARABIC_FONT_NAME, font_size=14):
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        canvas_obj.setFont(font, font_size)
        canvas_obj.drawString(x, y, bidi_text)
    except Exception:
        canvas_obj.setFont(font, font_size)
        canvas_obj.drawString(x, y, text)

def create_cover_pdf(title, answer, output_path):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    # Draw the title (Unicode/Arabic safe)
    draw_unicode_text(c, title, 50, height - 50, font=ARABIC_FONT_NAME, font_size=16)
    y = height - 90
    # Draw each line of answer/text (Unicode/Arabic safe)
    for line in textwrap.wrap(answer, 100):
        draw_unicode_text(c, line, 50, y, font=ARABIC_FONT_NAME, font_size=14)
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50
            if y < 50:
                c.showPage()
                y = height - 50
    c.save()

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
            ("LayoutLM OCR", layoutlm_ocr(pdf_path)),
            ("EasyOCR", easyocr_extract(pdf_path)),
        ]
        answers = {}
        cover_pdfs = []
        if os.path.exists(pdf_path):
            for method_name, extracted_text in extraction_methods:
                answers[method_name] = {}
                for q_idx, (question_name, question_template) in enumerate(QUESTIONS):
                    prompt = question_template.format(doc=extracted_text)
                    for model_name, model_id in OLLAMA_MODELS:
                        answer = query_ollama(model_id, prompt)
                        page_title = f"{method_name} | {model_name} | {question_name}"
                        cover_pdf_path = os.path.join(OUTPUT_DIR, f"cover_{pdf_id}_{method_name}_{model_name}_{q_idx}.pdf")
                        create_cover_pdf(page_title, answer, cover_pdf_path)
                        cover_pdfs.append(cover_pdf_path)
                        answers[method_name][f"{model_name}_{question_name}"] = answer
                # Add a page with the full extracted text
                text_page_title = f"{method_name} | Full Extracted Text"
                text_page_path = os.path.join(OUTPUT_DIR, f"cover_{pdf_id}_{method_name}_full_text.pdf")
                create_cover_pdf(text_page_title, extracted_text, text_page_path)
                cover_pdfs.append(text_page_path)
            # Merge all covers and original PDF
            try:
                writer = PdfWriter()
                for cover_pdf_path in cover_pdfs:
                    reader = PdfReader(cover_pdf_path)
                    for page in reader.pages:
                        writer.add_page(page)
                orig_reader = PdfReader(pdf_path)
                for page in orig_reader.pages:
                    writer.add_page(page)
                output_pdf_path = os.path.join(OUTPUT_DIR, f"llm_extraction_{pdf_id}_{pdf_filename}")
                with open(output_pdf_path, 'wb') as f_out:
                    writer.write(f_out)
                print(f"[SUCCESS] Output PDF created: {output_pdf_path}")
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
