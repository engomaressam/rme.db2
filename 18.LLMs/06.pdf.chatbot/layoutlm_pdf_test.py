import os
os.environ['PATH'] = r'C:\Program Files\poppler\Library\bin;' + os.environ['PATH']
from pdf2image import convert_from_path
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification
from PIL import Image
import torch
import pymysql
import random
from pytesseract import image_to_string, pytesseract
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Set Tesseract executable path
pytesseract.tesseract_cmd = r'C:\Users\Omar Essam2\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# MySQL connection settings
HOST = '10.10.11.242'
USER = 'omar2'
PASSWORD = 'Omar_54321'
DB = 'RME_TEST'
TABLE = 'po.pdfs'

# Load LayoutLM model and tokenizer
model_name = 'microsoft/layoutlm-base-uncased'
tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
model = LayoutLMForTokenClassification.from_pretrained(model_name)

def extract_layoutlm_from_pdf(pdf_path):
    print(f'Processing: {pdf_path}')
    images = convert_from_path(pdf_path, poppler_path=r'C:\Program Files\poppler\Library\bin')
    all_text = []
    for i, img in enumerate(images):
        text = image_to_string(img, lang='ara+eng')
        print(f'Page {i+1} OCR preview:\n{text[:300]}')
        all_text.append(text)
    return '\n'.join(all_text)

def get_random_pdf_paths(n=5):
    conn = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=DB, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    with conn.cursor() as cursor:
        cursor.execute(f"SELECT pdf_path FROM `{TABLE}` WHERE pdf_path IS NOT NULL AND pdf_path != '' ORDER BY RAND() LIMIT {n}")
        rows = cursor.fetchall()
    conn.close()
    return [row['pdf_path'] for row in rows]

if __name__ == '__main__':
    pdf_paths = get_random_pdf_paths(5)
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            text = extract_layoutlm_from_pdf(pdf_path)
            print(f'\n===== Extracted text from {os.path.basename(pdf_path)} =====\n{text[:1000]}\n')
        else:
            print(f'File not found: {pdf_path}')
