import os
import sys
import time
import logging
import tkinter as tk
from tkinter import filedialog
import mysql.connector
from mysql.connector import Error
import pytesseract
from pdf2image import convert_from_path
import hashlib

# Configure logging with UTF-8 encoding
class UTF8FileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)

# Configure logging to handle Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8FileHandler('pdf_extraction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set the path to Poppler binaries
poppler_path = r'C:\poppler\Library\bin'

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

class PDFExtractor:
    def __init__(self):
        self.folder_path = ""
        self.db_connection = None
        self.cursor = None
        self.processed_files = set()
    
    def connect_to_database(self):
        """Establish connection to MySQL database"""
        try:
            self.db_connection = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.db_connection.cursor()
            logging.info("Connected to MySQL database")
            
            # Create table if it doesn't exist
            self.create_table()
            
            # Load already processed files
            self.load_processed_files()
            
            return True
        except Error as e:
            logging.error(f"Error connecting to MySQL database: {e}")
            return False
    
    def create_table(self):
        """Create the po.pdfs table if it doesn't exist"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS `po.pdfs` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pdf_path VARCHAR(1000) NOT NULL,
            pdf_filename VARCHAR(255) NOT NULL,
            pdf_hash VARCHAR(64) NOT NULL,
            extracted_text LONGTEXT,
            processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (pdf_hash)
        )
        """
        
        # Create chunks table
        create_chunks_table_query = """
        CREATE TABLE IF NOT EXISTS `po.pdf_chunks` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pdf_id INT NOT NULL,
            chunk_number INT NOT NULL,
            chunk_text LONGTEXT,
            FOREIGN KEY (pdf_id) REFERENCES `po.pdfs`(id) ON DELETE CASCADE,
            INDEX (pdf_id, chunk_number)
        )
        """
        
        try:
            self.cursor.execute(create_table_query)
            self.cursor.execute(create_chunks_table_query)
            self.db_connection.commit()
            logging.info("Database tables created or already exist")
        except Error as e:
            logging.error(f"Error creating tables: {e}")
    
    def load_processed_files(self):
        """Load already processed files from the database"""
        query = "SELECT pdf_hash FROM `po.pdfs`"
        try:
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            self.processed_files = set([result[0] for result in results])
            logging.info(f"Loaded {len(self.processed_files)} already processed files")
        except Error as e:
            logging.error(f"Error loading processed files: {e}")
    
    def calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def browse_folder(self):
        """Open a file dialog to browse for a folder"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        folder_path = filedialog.askdirectory(title="Select Folder Containing PDF Files")
        
        if not folder_path:
            logging.warning("No folder selected. Exiting.")
            return False
        
        self.folder_path = folder_path
        logging.info(f"Selected folder: {folder_path}")
        return True
    
    def scan_pdfs(self, folder_path):
        """Scan a folder and all its subfolders for PDF files"""
        pdf_files = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(folder_path):
            # Find all PDF files in the current directory
            for file in files:
                if file.lower().endswith('.pdf'):
                    # Get the full path to the PDF file
                    full_path = os.path.join(root, file)
                    pdf_files.append(full_path)
        
        logging.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
        return pdf_files
    
    def split_text_into_chunks(self, text):
        """
        Split the extracted text into meaningful chunks (paragraphs).
        Skip single lines and merge them with adjacent paragraphs.
        """
        # Split text by double newlines (paragraph breaks)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Process paragraphs to handle single lines
        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            # If it's a single line (no newlines) and current_chunk is not empty
            if '\n' not in p and current_chunk:
                current_chunk += "\n" + p
            # If it's a single line and current_chunk is empty, start a new chunk
            elif '\n' not in p and not current_chunk:
                current_chunk = p
            # If it's a multi-line paragraph and current_chunk is not empty
            elif current_chunk:
                current_chunk += "\n" + p
                chunks.append(current_chunk)
                current_chunk = ""
            # If it's a multi-line paragraph and current_chunk is empty
            else:
                chunks.append(p)
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using Tesseract OCR with support for Arabic and English"""
        try:
            # Convert PDF to images using the specified poppler path
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
            
            # Extract text from each image with Arabic + English language support
            text = ""
            for image in images:
                # Use Arabic + English language configuration
                text += pytesseract.image_to_string(image, lang='ara+eng')
                text += "\n\n"
            
            return text
        except Exception as e:
            logging.error(f"Error in OCR processing: {str(e)}")
            raise
    
    def save_to_database(self, pdf_path, text, chunks):
        """Save extracted text and chunks to the database"""
        try:
            # Calculate file hash
            file_hash = self.calculate_file_hash(pdf_path)
            
            # Check if this file has already been processed
            if file_hash in self.processed_files:
                logging.info(f"Skipping already processed file: {pdf_path}")
                return
            
            # Insert main PDF record
            insert_pdf_query = """
            INSERT INTO `po.pdfs` (pdf_path, pdf_filename, pdf_hash, extracted_text)
            VALUES (%s, %s, %s, %s)
            """
            
            filename = os.path.basename(pdf_path)
            self.cursor.execute(insert_pdf_query, (pdf_path, filename, file_hash, text))
            pdf_id = self.cursor.lastrowid
            
            # Insert chunks
            insert_chunk_query = """
            INSERT INTO `po.pdf_chunks` (pdf_id, chunk_number, chunk_text)
            VALUES (%s, %s, %s)
            """
            
            for i, chunk in enumerate(chunks, 1):
                self.cursor.execute(insert_chunk_query, (pdf_id, i, chunk))
            
            # Commit the transaction
            self.db_connection.commit()
            
            # Add to processed files set
            self.processed_files.add(file_hash)
            
            logging.info(f"Saved PDF {filename} with {len(chunks)} chunks to database")
            
        except Error as e:
            logging.error(f"Database error: {e}")
            self.db_connection.rollback()
    
    def process_pdfs(self):
        """Process all PDF files in the selected folder"""
        if not self.connect_to_database():
            return
        
        if not self.browse_folder():
            return
        
        pdf_files = self.scan_pdfs(self.folder_path)
        
        if not pdf_files:
            logging.warning("No PDF files found in the selected directory.")
            return
        
        total_pdfs = len(pdf_files)
        processed = 0
        skipped = 0
        errors = 0
        
        for i, pdf_path in enumerate(pdf_files):
            try:
                # Calculate file hash
                file_hash = self.calculate_file_hash(pdf_path)
                
                # Skip if already processed
                if file_hash in self.processed_files:
                    safe_path = repr(pdf_path)
                    logging.info(f"Skipping already processed file ({i+1}/{total_pdfs}): {safe_path}")
                    skipped += 1
                    continue
                
                safe_path = repr(pdf_path)
                logging.info(f"Processing ({i+1}/{total_pdfs}): {safe_path}")
                
                # Extract text
                text = self.extract_text_from_pdf(pdf_path)
                
                # Split into chunks
                chunks = self.split_text_into_chunks(text)
                
                # Save to database
                self.save_to_database(pdf_path, text, chunks)
                
                processed += 1
                
                # Log progress
                if (i+1) % 5 == 0 or (i+1) == total_pdfs:
                    logging.info(f"Progress: {i+1}/{total_pdfs} PDFs processed")
                
            except Exception as e:
                safe_path = repr(pdf_path)
                logging.error(f"Error processing {safe_path}: {str(e)}")
                errors += 1
        
        # Close database connection
        if self.db_connection and self.db_connection.is_connected():
            self.cursor.close()
            self.db_connection.close()
            logging.info("Database connection closed")
        
        # Final summary
        logging.info(f"Processing complete: {processed} processed, {skipped} skipped, {errors} errors")

def main():
    extractor = PDFExtractor()
    extractor.process_pdfs()

if __name__ == "__main__":
    main()
