"""
Document Processor for GraphRAG Hybrid System
"""
import sys
import os
import re
import random
from loguru import logger
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

class DocumentProcessor:
    """
    Processes PDF documents for the GraphRAG system
    """
    
    def __init__(self, pdf_directory=None):
        """
        Initialize document processor
        
        Args:
            pdf_directory (str): Directory containing PDF files
        """
        self.pdf_directory = pdf_directory or config.PDF_DIRECTORY
        # Set Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF using OCR
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, poppler_path=r'C:\Program Files\poppler-24.08.0\Library\bin')
            
            all_text = ""
            for i, image in enumerate(images):
                # Perform OCR with support for multiple languages
                text = pytesseract.image_to_string(image, lang='eng+ara')
                all_text += text + "\n\n"
                logger.info(f"Extracted {len(text)} characters from page {i+1}")
            
            return all_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_po_number(self, text):
        """Extract PO number using regex patterns"""
        patterns = [
            r'P\.?O\.?\s*#?\s*(\w+[-/]?\w+)',
            r'Purchase\s+Order\s+(?:No\.?|Number)\s*[:.]?\s*(\w+[-/]?\w+)',
            r'Order\s+(?:No\.?|Number)\s*[:.]?\s*(\w+[-/]?\w+)',
            r'PO[-\s.:]*(\d+[-/]?\w*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for any alphanumeric string that looks like a PO number
        match = re.search(r'(?:^|\s)([A-Z0-9]{2,}[-/][A-Z0-9]{2,})', text)
        if match:
            return match.group(1).strip()
        
        return f"Unknown-PO-{random.randint(1000, 9999)}"
    
    def extract_project_info(self, text, folder_name=None):
        """
        Extract project name and code
        
        Args:
            text (str): Text to extract from
            folder_name (str): Folder name to use as project name if available
            
        Returns:
            dict: Project information
        """
        project_name = folder_name  # Use folder name as project name if provided
        project_code = None
        
        if not project_name:
            # Try to find project name in text
            project_patterns = [
                r'Project\s*(?:Name|Title)?\s*[:.]?\s*([A-Za-z0-9\s\-]+)',
                r'for\s+(?:the\s+)?project\s*[:.]?\s*([A-Za-z0-9\s\-]+)',
            ]
            
            for pattern in project_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    project_name = match.group(1).strip()
                    break
        
        # Try to find project code
        code_patterns = [
            r'Project\s*(?:Code|Number|ID|#)\s*[:.]?\s*([A-Za-z0-9\-]+)',
            r'Project\s*:\s*([A-Z0-9\-]+)',
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project_code = match.group(1).strip()
                break
        
        # If no project name found, use a generic one with the code if available
        if not project_name and project_code:
            project_name = f"Project {project_code}"
        elif not project_name:
            project_name = f"Unknown Project {random.randint(100, 999)}"
        
        return {
            "name": project_name,
            "code": project_code
        }
    
    def extract_supplier_info(self, text):
        """Extract supplier name and details"""
        supplier_name = None
        supplier_id = None
        
        # Try to find supplier name
        supplier_patterns = [
            r'(?:Vendor|Supplier|Company)(?:\s+Name)?\s*[:.]?\s*([A-Za-z0-9\s\-&\.]+)',
            r'(?:Bill|Ship)\s+To\s*[:.]?\s*([A-Za-z0-9\s\-&\.]+)',
            r'(?:^|\n)([A-Z][A-Za-z0-9\s\-&\.]+)(?:\n|$)',
        ]
        
        for pattern in supplier_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                supplier_name = match.group(1).strip()
                break
        
        # Try to find supplier ID
        id_patterns = [
            r'(?:Vendor|Supplier)\s*(?:Code|ID|Number|#)\s*[:.]?\s*([A-Za-z0-9\-]+)',
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                supplier_id = match.group(1).strip()
                break
        
        # If no supplier name found, use a generic one
        if not supplier_name:
            supplier_name = f"Unknown Supplier {random.randint(100, 999)}"
        
        return {
            "name": supplier_name,
            "id": supplier_id,
            "category": None
        }
    
    def extract_date(self, text):
        """Extract date from text"""
        date_patterns = [
            r'(?:Date|PO\s+Date|Order\s+Date)\s*[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:Date|PO\s+Date|Order\s+Date)\s*[:.]?\s*(\d{1,2}\s+[A-Za-z]+\s+\d{2,4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_items(self, text):
        """Extract items from text using regex patterns"""
        items = []
        
        # Try to find items in a table-like structure
        # This is a simplified approach and might need adjustment based on actual PO formats
        item_patterns = [
            r'(\d+)\s+([A-Za-z0-9\s\-\.]+)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\s+(\d+(?:\.\d+)?)',
            r'([A-Za-z0-9\s\-\.]+)\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\s+(\d+(?:\.\d+)?)',
        ]
        
        for pattern in item_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) == 5:  # With item number
                    _, name, quantity, unit, value = match.groups()
                else:  # Without item number
                    name, quantity, unit, value = match.groups()
                
                try:
                    items.append({
                        "name": name.strip(),
                        "quantity": float(quantity),
                        "unit": unit.strip(),
                        "value": float(value)
                    })
                except ValueError:
                    # Skip if conversion to float fails
                    pass
        
        # If no items found, add a dummy item
        if not items:
            items.append({
                "name": f"Unspecified Item {random.randint(100, 999)}",
                "quantity": 1,
                "unit": "Unit",
                "value": 1000
            })
        
        return items
    
    def extract_total_value(self, text):
        """Extract total value from text"""
        total_patterns = [
            r'(?:Total|Grand\s+Total|Amount)\s*[:.]?\s*(?:USD|SAR|$|£|€)?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(?:Total|Grand\s+Total|Amount)\s*[:.]?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:USD|SAR|$|£|€)?',
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Remove commas and convert to float
                try:
                    return float(match.group(1).replace(',', ''))
                except ValueError:
                    pass
        
        # If no total found, return None
        return None
    
    def extract_po_details(self, text, folder_name=None):
        """
        Extract purchase order details from text
        
        Args:
            text (str): Text to extract from
            folder_name (str): Folder name to use as project name if available
            
        Returns:
            dict: Purchase order details
        """
        po_data = {
            "po_number": self.extract_po_number(text),
            "project": self.extract_project_info(text, folder_name),
            "date": self.extract_date(text),
            "supplier": self.extract_supplier_info(text),
            "items": self.extract_items(text),
            "total_value": self.extract_total_value(text),
            "payment_terms": None,
            "delivery_terms": None
        }
        
        # If total_value is not found, calculate from items
        if po_data["total_value"] is None and po_data["items"]:
            po_data["total_value"] = sum(item["value"] for item in po_data["items"])
        
        return po_data
    
    def process_pdf_file(self, pdf_path, folder_name=None):
        """
        Process a PDF file and extract purchase order details
        
        Args:
            pdf_path (str): Path to PDF file
            folder_name (str): Folder name to use as project name if available
            
        Returns:
            dict: Purchase order details
        """
        try:
            logger.info(f"Processing PDF file: {pdf_path}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                logger.error(f"No text extracted from {pdf_path}")
                return None
            
            # Extract purchase order details
            po_data = self.extract_po_details(text, folder_name)
            logger.info(f"Extracted PO: {po_data['po_number']}")
            logger.info(f"Project: {po_data['project']['name']}")
            logger.info(f"Supplier: {po_data['supplier']['name']}")
            logger.info(f"Items: {len(po_data['items'])}")
            
            return po_data
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            return None
    
    def process_directory(self, directory=None, recursive=True):
        """
        Process all PDF files in a directory
        
        Args:
            directory (str): Directory to process
            recursive (bool): Process subdirectories
            
        Returns:
            list: List of purchase order details
        """
        directory = directory or self.pdf_directory
        po_data_list = []
        
        try:
            logger.info(f"Processing directory: {directory}")
            
            # Get folder name for project name
            folder_name = os.path.basename(directory)
            
            # Process files in the current directory
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isfile(item_path) and item.lower().endswith('.pdf'):
                    # Process PDF file
                    po_data = self.process_pdf_file(item_path, folder_name)
                    if po_data:
                        po_data_list.append(po_data)
                elif os.path.isdir(item_path) and recursive:
                    # Process subdirectory
                    sub_po_data_list = self.process_directory(item_path, recursive)
                    po_data_list.extend(sub_po_data_list)
            
            logger.info(f"Processed {len(po_data_list)} purchase orders in {directory}")
            return po_data_list
        except Exception as e:
            logger.error(f"Error processing directory: {str(e)}")
            return po_data_list

if __name__ == "__main__":
    # Test document processor
    processor = DocumentProcessor()
    print(f"PDF directory: {processor.pdf_directory}")
    
    # Process a single PDF file
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
            po_data = processor.process_pdf_file(pdf_path)
            if po_data:
                print(f"Extracted PO: {po_data['po_number']}")
                print(f"Project: {po_data['project']['name']}")
                print(f"Supplier: {po_data['supplier']['name']}")
                print(f"Items: {len(po_data['items'])}")
        else:
            print(f"Invalid PDF file: {pdf_path}")
    else:
        print("No PDF file specified. Use: python document_processor.py <pdf_file>")
