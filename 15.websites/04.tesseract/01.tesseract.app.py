from flask import Flask, render_template, request
import PyPDF2
from PIL import Image
import pytesseract
import io
import os
import tempfile
from pdf2image import convert_from_path

# Set the path to poppler
os.environ['PATH'] += os.pathsep + r'C:\poppler\Library\bin'

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', result=None)

@app.route('/extract', methods=['POST'])
def extract_text():
    if 'pdf_file' not in request.files:
        return render_template('index.html', result='No file uploaded')
    
    file = request.files['pdf_file']
    if file.filename == '':
        return render_template('index.html', result='No file selected')
    
    extraction_method = request.form.get('extraction_method', 'pypdf')
    ocr_language = request.form.get('ocr_language', 'eng')  # Default to English if not specified
    
    try:
        # Save the uploaded file to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        file.save(temp_file.name)
        temp_file.close()
        
        if extraction_method == 'pypdf':
            # Extract text using PyPDF2
            extracted_text = []
            with open(temp_file.name, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text:
                        extracted_text.append(text)
                    else:
                        extracted_text.append(f"[Page {page_num+1} contains no extractable text]")
            
            result = '\n\n--- Page Break ---\n\n'.join(extracted_text)
            method_used = "PyPDF2 Text Extraction"
            
        elif extraction_method == 'tesseract':
            # Use pdf2image and Tesseract OCR
            try:
                # Convert PDF to images using pdf2image with poppler path
                images = convert_from_path(temp_file.name)
                
                # Extract text using Tesseract OCR
                extracted_text = []
                for i, image in enumerate(images):
                    # Convert image to grayscale for better OCR
                    image = image.convert('L')
                    
                    # Use Tesseract to extract text with the selected language
                    text = pytesseract.image_to_string(image, lang=ocr_language)
                    
                    if text.strip():
                        extracted_text.append(text)
                    else:
                        extracted_text.append(f"[Page {i+1} contains no extractable text]")
                
                result = '\n\n--- Page Break ---\n\n'.join(extracted_text)
                method_used = "Tesseract OCR"
                
            except Exception as e:
                result = f"OCR Error: {str(e)}"
                method_used = "Error"
        else:
            result = "Invalid extraction method"
            method_used = "Error"
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        return render_template('index.html', result=f"[Extraction Method: {method_used}]\n\n{result}")
        
    except Exception as e:
        return render_template('index.html', result=f'Error processing PDF: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)