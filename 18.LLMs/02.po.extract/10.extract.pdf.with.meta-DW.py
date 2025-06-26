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
import re

# Configure logging with UTF-8 encoding
class UTF8FileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)

# Configure logging to handle Unicode characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8FileHandler('pdf_extraction_with_meta.log'),
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

# Base path for projects
BASE_PATH = r'\\fileserver2\Head Office Server\Procurement (PR)\02 Projects Document\02-01 Finish'

# Document type keywords
CONTRACT_KEYWORDS = ['contract', 'contracts']
COMPARISON_KEYWORDS = ['comparison', 'comparisons', 'comparision']
PO_KEYWORDS = ['lpo', 'po', 'purchase', 'purchasing', 'order', 'tpd']

class PDFExtractorWithMeta:
    def __init__(self):
        self.folder_path = ""
        self.db_connection = None
        self.cursor = None
        self.processed_files = set()
        self.project_metadata = {}
    
    def connect_to_database(self):
        """Establish connection to MySQL database"""
        try:
            self.db_connection = mysql.connector.connect(**DB_CONFIG)
            self.cursor = self.db_connection.cursor()
            logging.info("Connected to MySQL database")
            
            # Create tables if they don't exist
            self.create_tables()
            
            # Load already processed files
            self.load_processed_files()
            
            return True
        except Error as e:
            logging.error(f"Error connecting to MySQL database: {e}")
            return False
    
    def create_tables(self):
        """Create the necessary tables if they don't exist"""
        # Create main PDF table with metadata columns
        create_table_query = """
        CREATE TABLE IF NOT EXISTS `po.pdfs` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pdf_path VARCHAR(1000) NOT NULL,
            pdf_filename VARCHAR(255) NOT NULL,
            pdf_hash VARCHAR(64) NOT NULL,
            extracted_text LONGTEXT,
            processed_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            project_name VARCHAR(255),
            document_type VARCHAR(50),
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
        
        # Create project summary table
        create_summary_table_query = """
        CREATE TABLE IF NOT EXISTS `po.project_summary` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            project_name VARCHAR(255) NOT NULL,
            document_type VARCHAR(50) NOT NULL,
            pdf_count INT NOT NULL,
            UNIQUE KEY (project_name, document_type)
        )
        """
        
        try:
            self.cursor.execute(create_table_query)
            self.cursor.execute(create_chunks_table_query)
            self.cursor.execute(create_summary_table_query)
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
    
    def determine_document_type(self, folder_name):
        """Determine document type based on folder name"""
        folder_lower = folder_name.lower()
        
        # Check for contract
        for keyword in CONTRACT_KEYWORDS:
            if keyword in folder_lower:
                return 'Contract'
        
        # Check for comparison
        for keyword in COMPARISON_KEYWORDS:
            if keyword in folder_lower:
                return 'Comparison'
        
        # Check for PO
        for keyword in PO_KEYWORDS:
            if keyword in folder_lower:
                return 'Purchase Order'
        
        # Default if no match
        return 'Unknown'
    
    def scan_project_structure(self):
        """Scan the project structure and return a dictionary of project metadata"""
        project_metadata = {}
        
        try:
            # Get all project folders
            project_folders = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]
            logging.info(f"Found {len(project_folders)} project folders")
            
            for project_folder in project_folders:
                project_path = os.path.join(BASE_PATH, project_folder)
                
                # Get document type folders within the project
                try:
                    document_folders = [f for f in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, f))]
                    
                    for doc_folder in document_folders:
                        doc_type = self.determine_document_type(doc_folder)
                        doc_path = os.path.join(project_path, doc_folder)
                        
                        # Walk through all files in this document folder
                        for root, dirs, files in os.walk(doc_path):
                            for file in files:
                                if file.lower().endswith('.pdf'):
                                    full_path = os.path.join(root, file)
                                    
                                    # Store metadata
                                    project_metadata[full_path] = {
                                        'project_name': project_folder,
                                        'document_type': doc_type
                                    }
                except Exception as e:
                    logging.error(f"Error scanning project {project_folder}: {e}")
                    continue
            
            logging.info(f"Identified {len(project_metadata)} PDF files with metadata")
            self.project_metadata = project_metadata
            return project_metadata
        except Exception as e:
            logging.error(f"Error scanning project structure: {e}")
            return {}
    
    def get_metadata_for_pdf(self, pdf_path):
        """Get metadata for a PDF file based on its path"""
        # Normalize path (handle different path formats)
        normalized_path = pdf_path.replace('/', '\\')
        
        # Try to find metadata for this path
        metadata = None
        
        # Direct match
        if normalized_path in self.project_metadata:
            metadata = self.project_metadata[normalized_path]
        else:
            # Try to find a partial match
            for path, meta in self.project_metadata.items():
                # Extract the project name and file name parts
                if os.path.basename(normalized_path) == os.path.basename(path):
                    # Check if the path contains the project name
                    if meta['project_name'] in normalized_path:
                        metadata = meta
                        break
        
        # Return default metadata if not found
        if not metadata:
            return {
                'project_name': 'Unknown',
                'document_type': 'Unknown'
            }
        
        return metadata
    
    def save_to_database(self, pdf_path, text, chunks):
        """Save extracted text, chunks, and metadata to the database"""
        try:
            # Calculate file hash
            file_hash = self.calculate_file_hash(pdf_path)
            
            # Check if this file has already been processed
            if file_hash in self.processed_files:
                logging.info(f"Skipping already processed file: {pdf_path}")
                return
            
            # Get metadata for this PDF
            metadata = self.get_metadata_for_pdf(pdf_path)
            
            # Insert main PDF record with metadata
            insert_pdf_query = """
            INSERT INTO `po.pdfs` (pdf_path, pdf_filename, pdf_hash, extracted_text, project_name, document_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            filename = os.path.basename(pdf_path)
            self.cursor.execute(insert_pdf_query, (
                pdf_path, 
                filename, 
                file_hash, 
                text, 
                metadata['project_name'], 
                metadata['document_type']
            ))
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
            
            logging.info(f"Saved PDF {filename} with {len(chunks)} chunks to database (Project: {metadata['project_name']}, Type: {metadata['document_type']})")
            
        except Error as e:
            logging.error(f"Database error: {e}")
            self.db_connection.rollback()
    
    def update_project_summary(self):
        """Update the project summary table"""
        try:
            # Clear existing data
            self.cursor.execute("TRUNCATE TABLE `po.project_summary`")
            
            # Insert summary data
            self.cursor.execute("""
            INSERT INTO `po.project_summary` (project_name, document_type, pdf_count)
            SELECT 
                project_name, 
                document_type, 
                COUNT(*) as pdf_count
            FROM `po.pdfs`
            WHERE project_name IS NOT NULL AND document_type IS NOT NULL
            GROUP BY project_name, document_type
            ORDER BY project_name, document_type
            """)
            
            self.db_connection.commit()
            
            # Get summary counts
            self.cursor.execute("SELECT COUNT(*) FROM `po.project_summary`")
            summary_count = self.cursor.fetchone()[0]
            
            self.cursor.execute("SELECT COUNT(DISTINCT project_name) FROM `po.project_summary`")
            project_count = self.cursor.fetchone()[0]
            
            logging.info(f"Updated project summary table with {summary_count} entries across {project_count} projects")
            
        except Error as e:
            logging.error(f"Error updating project summary: {e}")
    
    def process_pdfs(self):
        """Process all PDF files in the selected folder"""
        if not self.connect_to_database():
            return
        
        # Scan project structure to get metadata
        logging.info("Scanning project structure for metadata...")
        self.scan_project_structure()
        
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
                
                # Save to database with metadata
                self.save_to_database(pdf_path, text, chunks)
                
                processed += 1
                
                # Log progress
                if (i+1) % 5 == 0 or (i+1) == total_pdfs:
                    logging.info(f"Progress: {i+1}/{total_pdfs} PDFs processed")
                
            except Exception as e:
                safe_path = repr(pdf_path)
                logging.error(f"Error processing {safe_path}: {str(e)}")
                errors += 1
        
        # Update project summary table
        logging.info("Updating project summary table...")
        self.update_project_summary()
        
        # Close database connection
        if self.db_connection and self.db_connection.is_connected():
            self.cursor.close()
            self.db_connection.close()
            logging.info("Database connection closed")
        
        # Final summary
        logging.info(f"Processing complete: {processed} processed, {skipped} skipped, {errors} errors")

def main():
    logging.info("Starting PDF extraction with metadata process...")
    extractor = PDFExtractorWithMeta()
    extractor.process_pdfs()

if __name__ == "__main__":
    main()
