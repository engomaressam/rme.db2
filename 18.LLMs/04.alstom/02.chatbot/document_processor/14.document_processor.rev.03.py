"""
Document Processor for Alstom Project Assistant
This script extracts text from PDF documents, creates embeddings, and stores them in a vector database.
It is separate from the web interface to allow for more efficient processing and better error handling.
"""

import os
import json
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("document_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DocumentProcessor")

# Define paths
# Updated to use network path instead of local path
BASE_PATH = "\\\\fileserver2\\Head Office Server\\Projects Control (PC)\\10 Backup\\05 Models\\alstom"
VECTOR_DB_DIR = "C:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\05.llm\\alstom"
DOCUMENTS_DIR = os.path.join(VECTOR_DB_DIR, "documents")
EMBEDDINGS_FILE = os.path.join(VECTOR_DB_DIR, "document_embeddings.json")
DOCUMENTS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "document_index.json")

# Create necessary directories
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Create .gitignore file
def create_gitignore():
    gitignore_path = os.path.join(VECTOR_DB_DIR, ".gitignore")
    if not os.path.exists(gitignore_path):
        with open(gitignore_path, 'w') as f:
            f.write("# Ignore vector database files\n")
            f.write("document_embeddings.json\n")
            f.write("document_index.json\n")
            f.write("# Ignore document files\n")
            f.write("documents/\n")
        logger.info(f"Created .gitignore file at {gitignore_path}")

# Initialize embedding model
def initialize_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
        # Using a smaller but effective model for embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Successfully loaded SentenceTransformer model")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        try:
            # Try installing if not available
            import subprocess
            logger.info("Attempting to install sentence-transformers...")
            subprocess.check_call(["pip", "install", "sentence-transformers"])
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully installed and loaded SentenceTransformer model")
            return model
        except Exception as e2:
            logger.error(f"Failed to install sentence-transformers: {str(e2)}")
            return None

# Extract text from PDF using PyPDF2
def extract_text_with_pypdf2(pdf_path):
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed for {pdf_path}: {str(e)}")
        return ""

# Check if Tesseract OCR is installed
def check_tesseract_installation():
    try:
        import pytesseract
        import shutil
        
        # Check if tesseract executable exists in PATH
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            logger.info(f"Tesseract OCR found at: {tesseract_path}")
            return True
        else:
            # Try to use the default Windows installation path
            default_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(default_path):
                pytesseract.pytesseract.tesseract_cmd = default_path
                logger.info(f"Tesseract OCR found at default location: {default_path}")
                return True
            else:
                logger.warning("Tesseract OCR not found in PATH or default location")
                logger.warning("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
                logger.warning("After installation, set the path to tesseract.exe:")
                logger.warning("pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
                return False
    except Exception as e:
        logger.error(f"Error checking Tesseract installation: {str(e)}")
        return False

# Extract text from PDF using Tesseract OCR
def extract_text_with_tesseract(pdf_path):
    try:
        import pytesseract
        from pdf2image import convert_from_path
        
        # Check Tesseract installation
        if not hasattr(extract_text_with_tesseract, 'tesseract_checked'):
            extract_text_with_tesseract.tesseract_checked = check_tesseract_installation()
            if not extract_text_with_tesseract.tesseract_checked:
                return ""
        
        logger.info(f"Using Tesseract OCR for {pdf_path}")
        
        # Use the specific Poppler path
        poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"
        
        try:
            # Try with the specific poppler path
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
            logger.info(f"Successfully converted PDF to images using Poppler at {poppler_path}")
        except Exception as poppler_error:
            logger.warning(f"Error with specified poppler path: {str(poppler_error)}")
            
            # Try with default path as fallback
            try:
                images = convert_from_path(pdf_path)
                logger.info("Successfully converted PDF to images using default Poppler")
            except Exception as default_error:
                logger.error(f"All PDF conversion methods failed: {str(default_error)}")
                return ""
        
        text = ""
        for i, image in enumerate(images):
            text += pytesseract.image_to_string(image) + "\n"
            
        return text.strip()
    except Exception as e:
        logger.error(f"Tesseract OCR extraction failed for {pdf_path}: {str(e)}")
        return ""

# Extract text from PDF with fallback to OCR if needed
def extract_pdf_content(pdf_path):
    # First try PyPDF2
    text = extract_text_with_pypdf2(pdf_path)
    
    # If text extraction failed or returned very little text, try Tesseract OCR
    if not text or len(text.split()) < 20:
        logger.info(f"PyPDF2 extraction yielded insufficient text ({len(text.split())} words), trying Tesseract OCR")
        text = extract_text_with_tesseract(pdf_path)
    
    # If both methods failed, log an error
    if not text:
        logger.error(f"All text extraction methods failed for {pdf_path}")
    else:
        logger.info(f"Successfully extracted {len(text.split())} words from {pdf_path}")
    
    return text

# Scan directory for PDF files
def scan_directory(base_path):
    logger.info(f"Scanning {base_path} for PDF files...")
    pdf_files = []
    
    if not os.path.exists(base_path):
        logger.error(f"Directory {base_path} does not exist")
        return pdf_files
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                full_path = os.path.join(root, file)
                # Create a simplified path relative to the base path
                rel_path = os.path.relpath(os.path.dirname(full_path), base_path)
                
                pdf_files.append({
                    "name": file,
                    "path": full_path,
                    "simplified_path": rel_path,
                    "last_modified": os.path.getmtime(full_path)
                })
    
    logger.info(f"Found {len(pdf_files)} PDF files in {base_path}")
    return pdf_files

# Copy PDF to documents directory
def copy_pdf_to_documents_dir(pdf_info):
    try:
        # Create subdirectory structure matching the original
        target_dir = os.path.join(DOCUMENTS_DIR, pdf_info["simplified_path"])
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy the file
        target_path = os.path.join(target_dir, pdf_info["name"])
        shutil.copy2(pdf_info["path"], target_path)
        
        logger.info(f"Copied {pdf_info['path']} to {target_path}")
        return target_path
    except Exception as e:
        logger.error(f"Error copying {pdf_info['path']}: {str(e)}")
        return None

# Process documents and create embeddings
def process_documents(embedding_model, force_reprocess=False):
    # Load existing document index if it exists
    document_index = {}
    if os.path.exists(DOCUMENTS_INDEX_FILE) and not force_reprocess:
        try:
            with open(DOCUMENTS_INDEX_FILE, 'r', encoding='utf-8') as f:
                document_index = json.load(f)
            logger.info(f"Loaded existing document index with {len(document_index)} documents")
        except Exception as e:
            logger.error(f"Error loading document index: {str(e)}")
            document_index = {}
    
    # Scan for PDF files
    pdf_files = scan_directory(BASE_PATH)
    
    # Process each PDF file
    for pdf_info in tqdm(pdf_files, desc="Processing documents"):
        doc_id = f"{pdf_info['name']}_{pdf_info['simplified_path']}"
        
        # Check if document already processed and hasn't changed
        if (not force_reprocess and 
            doc_id in document_index and 
            'last_modified' in document_index[doc_id] and
            pdf_info['last_modified'] <= document_index[doc_id]['last_modified']):
            logger.info(f"Skipping already processed document: {pdf_info['name']}")
            continue
        
        logger.info(f"Processing document: {pdf_info['name']}")
        
        # Copy PDF to documents directory
        local_path = copy_pdf_to_documents_dir(pdf_info)
        
        # Extract text content
        text_content = extract_pdf_content(pdf_info["path"])
        
        # Update document index
        document_index[doc_id] = {
            "name": pdf_info["name"],
            "path": pdf_info["path"],
            "local_path": local_path,
            "simplified_path": pdf_info["simplified_path"],
            "last_modified": pdf_info["last_modified"],
            "processed_date": datetime.now().isoformat(),
            "word_count": len(text_content.split()),
            "content": text_content
        }
    
    # Save document index
    try:
        with open(DOCUMENTS_INDEX_FILE, 'w', encoding='utf-8') as f:
            json.dump(document_index, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved document index with {len(document_index)} documents")
    except Exception as e:
        logger.error(f"Error saving document index: {str(e)}")
    
    return document_index

# Create embeddings for documents
def create_embeddings(embedding_model, document_index):
    if embedding_model is None:
        logger.error("Cannot create embeddings: model not loaded")
        return {}
    
    logger.info("Creating document embeddings...")
    embeddings = {}
    
    for doc_id, doc_info in tqdm(document_index.items(), desc="Creating embeddings"):
        # Create a rich text representation of the document
        doc_text = f"Document: {doc_info['name']}\nPath: {doc_info['simplified_path']}\n"
        
        # Add document content
        if 'content' in doc_info and doc_info['content']:
            doc_text += f"Content: {doc_info['content']}\n"
        
        # Generate embedding
        try:
            embedding = embedding_model.encode(doc_text)
            embeddings[doc_id] = embedding.tolist()  # Convert to list for JSON serialization
        except Exception as e:
            logger.error(f"Error embedding document {doc_info['name']}: {str(e)}")
    
    # Save embeddings
    try:
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f)
        logger.info(f"Saved embeddings for {len(embeddings)} documents")
    except Exception as e:
        logger.error(f"Error saving embeddings: {str(e)}")
    
    return embeddings

# Main function
def main():
    parser = argparse.ArgumentParser(description="Process documents and create embeddings for Alstom Project Assistant")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all documents")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--check-tesseract", action="store_true", help="Check Tesseract OCR installation")
    args = parser.parse_args()
    
    # Create .gitignore file
    create_gitignore()
    
    # Check Tesseract installation if requested
    if args.check_tesseract:
        is_installed = check_tesseract_installation()
        if not is_installed:
            logger.warning("\n\n==== TESSERACT OCR INSTALLATION INSTRUCTIONS ====\n")
            logger.warning("1. Download Tesseract OCR installer from: https://github.com/UB-Mannheim/tesseract/wiki")
            logger.warning("2. Run the installer and complete the installation")
            logger.warning("3. Download poppler for Windows from: https://github.com/oschwartz10612/poppler-windows/releases/")
            logger.warning("4. Extract the poppler zip file and add the bin directory to your PATH")
            logger.warning("5. Restart your command prompt or IDE")
            logger.warning("\nAfter installation, run this script again without the --check-tesseract flag\n")
            return
    
    # Install dependencies if requested
    if args.install_deps:
        import subprocess
        logger.info("Installing dependencies...")
        dependencies = [
            "sentence-transformers",
            "PyPDF2",
            "pytesseract",
            "pdf2image",
            "tqdm",
            "numpy"
        ]
        for dep in dependencies:
            try:
                logger.info(f"Installing {dep}...")
                subprocess.check_call(["pip", "install", dep])
            except Exception as e:
                logger.error(f"Error installing {dep}: {str(e)}")
    
    # Initialize embedding model
    embedding_model = initialize_embedding_model()
    if embedding_model is None:
        logger.error("Failed to initialize embedding model. Exiting.")
        return
    
    # Process documents
    document_index = process_documents(embedding_model, force_reprocess=args.force)
    
    # Create embeddings
    embeddings = create_embeddings(embedding_model, document_index)
    
    logger.info("Document processing complete!")
    logger.info(f"Processed {len(document_index)} documents")
    logger.info(f"Created {len(embeddings)} embeddings")

if __name__ == "__main__":
    # Print banner
    print("\n" + "=" * 80)
    print("Alstom Project Document Processor (v1.0)")
    print("This script extracts text from PDF documents and creates embeddings for the vector database.")
    print("=" * 80)
    print("\nIMPORTANT: This script requires Tesseract OCR to be installed for optimal text extraction.")
    print("If you haven't installed Tesseract OCR yet, run: python 14.document_processor.py --check-tesseract")
    print("\n")
    
    main()
