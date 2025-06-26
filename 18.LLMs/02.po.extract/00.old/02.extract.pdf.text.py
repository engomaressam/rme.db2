import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
import threading
import queue

# Set the path to Poppler binaries
poppler_path = r'C:\poppler\Library\bin'

# Set the path to Tesseract executable if it's not in your PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PDFTextExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Text Extractor")
        self.root.geometry("800x600")
        
        # Create a queue for communication between threads
        self.queue = queue.Queue()
        
        # Create main frame
        self.main_frame = tk.Frame(root, padx=20, pady=20)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header_label = tk.Label(self.main_frame, text="PDF Text Extractor", font=("Arial", 16, "bold"))
        header_label.pack(pady=10)
        
        # Create description
        desc_label = tk.Label(self.main_frame, text="Extract text from PDF files using Tesseract OCR and save to Excel")
        desc_label.pack(pady=5)
        
        # Create button frame
        button_frame = tk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        # Create browse button
        self.browse_button = tk.Button(button_frame, text="Select Folder", command=self.browse_folder, width=15)
        self.browse_button.pack(side=tk.LEFT, padx=10)
        
        # Create extract button
        self.extract_button = tk.Button(button_frame, text="Extract Text", command=self.start_extraction, width=15, state=tk.DISABLED)
        self.extract_button.pack(side=tk.LEFT, padx=10)
        
        # Create status frame
        self.status_frame = tk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=10)
        
        # Create status label
        self.status_label = tk.Label(self.status_frame, text="Status: Ready", anchor=tk.W)
        self.status_label.pack(fill=tk.X)
        
        # Create progress frame
        self.progress_frame = tk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        # Create progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(self.progress_frame, variable=self.progress_var, 
                                    from_=0, to=100, orient=tk.HORIZONTAL, 
                                    state=tk.DISABLED, showvalue=True, label="Progress")
        self.progress_bar.pack(fill=tk.X)
        
        # Create log frame
        log_frame = tk.Frame(self.main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create log label
        log_label = tk.Label(log_frame, text="Processing Log:", anchor=tk.W)
        log_label.pack(anchor=tk.W)
        
        # Create scrollbar for log
        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create log text area
        self.log_text = tk.Text(log_frame, height=15, width=70, yscrollcommand=scrollbar.set)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.log_text.yview)
        
        # Folder path variable
        self.folder_path = ""
        self.pdf_files = []
        
        # Start the queue processing
        self.process_queue()
        
    def browse_folder(self):
        """Open a file dialog to browse for a folder"""
        folder_path = filedialog.askdirectory(title="Select Folder Containing PDF Files")
        
        if not folder_path:
            return
            
        self.folder_path = folder_path
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, f"Selected folder: {folder_path}\n")
        
        # Scan for PDFs
        self.status_label.config(text="Status: Scanning for PDF files...")
        self.pdf_files = self.scan_pdfs(folder_path)
        
        if len(self.pdf_files) == 0:
            self.status_label.config(text="Status: No PDF files found")
            messagebox.showinfo("No PDFs Found", "No PDF files found in the selected directory or its subdirectories.")
        else:
            self.status_label.config(text=f"Status: Found {len(self.pdf_files)} PDF files")
            self.log_text.insert(tk.END, f"Found {len(self.pdf_files)} PDF files\n")
            self.extract_button.config(state=tk.NORMAL)
    
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
                    # Store the full path and the relative path from the base folder
                    rel_path = os.path.relpath(full_path, folder_path)
                    pdf_files.append((full_path, rel_path))
        
        return pdf_files
    
    def start_extraction(self):
        """Start the extraction process in a separate thread"""
        self.extract_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.progress_bar.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Extracting text from PDFs...")
        
        # Start extraction in a separate thread
        extraction_thread = threading.Thread(target=self.extract_text_from_pdfs)
        extraction_thread.daemon = True
        extraction_thread.start()
    
    def split_text_into_chunks(self, text):
        """
        Split the extracted text into meaningful chunks (paragraphs).
        Skip single lines and merge them with adjacent paragraphs.
        
        Args:
            text (str): The extracted text from the PDF
            
        Returns:
            list: List of text chunks
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
    
    def extract_text_from_pdfs(self):
        """Extract text from PDFs using Tesseract OCR and save to Excel"""
        total_pdfs = len(self.pdf_files)
        results = []
        max_chunks = 0  # Track the maximum number of chunks across all PDFs
        
        # First pass: Extract text and determine max chunks
        pdf_data = []
        
        for i, (full_path, rel_path) in enumerate(self.pdf_files):
            try:
                # Update progress
                progress = int((i / total_pdfs) * 100)
                self.queue.put(("progress", progress))
                self.queue.put(("log", f"Processing {i+1}/{total_pdfs}: {rel_path}\n"))
                
                # Extract text from PDF
                text = self.extract_text_from_pdf(full_path)
                
                # Split text into chunks
                chunks = self.split_text_into_chunks(text)
                
                # Update max chunks count
                max_chunks = max(max_chunks, len(chunks))
                
                # Store data for this PDF
                pdf_data.append({
                    'full_path': full_path,
                    'filename': os.path.basename(full_path),
                    'text': text,
                    'chunks': chunks
                })
                
                self.queue.put(("log", f"Completed: {rel_path} - Found {len(chunks)} text chunks\n"))
                
            except Exception as e:
                self.queue.put(("log", f"Error processing {rel_path}: {str(e)}\n"))
        
        # Second pass: Create results with dynamic columns for chunks
        for pdf in pdf_data:
            # Create base record
            record = {
                'PDF Path': pdf['full_path'],
                'PDF Filename': pdf['filename'],
                'Extracted Text': pdf['text']
            }
            
            # Add chunks as separate columns
            for i, chunk in enumerate(pdf['chunks'], 1):
                record[f'Chunk_{i}'] = chunk
            
            # Fill in empty chunks up to max_chunks
            for i in range(len(pdf['chunks']) + 1, max_chunks + 1):
                record[f'Chunk_{i}'] = ""
            
            results.append(record)
        
        # Save results to Excel
        if results:
            try:
                excel_path = os.path.join(self.folder_path, "extracted_text.xlsx")
                df = pd.DataFrame(results)
                df.to_excel(excel_path, index=False)
                self.queue.put(("log", f"Results saved to: {excel_path}\n"))
                self.queue.put(("status", f"Status: Completed. Results saved to Excel file"))
                self.queue.put(("complete", None))
            except Exception as e:
                self.queue.put(("log", f"Error saving to Excel: {str(e)}\n"))
                self.queue.put(("status", "Status: Error saving to Excel"))
                self.queue.put(("complete", None))
        else:
            self.queue.put(("status", "Status: No text was extracted"))
            self.queue.put(("complete", None))
    
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
            raise Exception(f"Error in OCR processing: {str(e)}")

    
    def process_queue(self):
        """Process messages from the queue"""
        try:
            while True:
                message_type, message = self.queue.get(block=False)
                
                if message_type == "progress":
                    self.progress_var.set(message)
                elif message_type == "log":
                    self.log_text.insert(tk.END, message)
                    self.log_text.see(tk.END)
                elif message_type == "status":
                    self.status_label.config(text=message)
                elif message_type == "complete":
                    self.progress_bar.config(state=tk.DISABLED)
                    self.browse_button.config(state=tk.NORMAL)
                    self.extract_button.config(state=tk.NORMAL)
                
                self.queue.task_done()
        except queue.Empty:
            # Queue is empty, schedule next check
            self.root.after(100, self.process_queue)

def main():
    root = tk.Tk()
    app = PDFTextExtractor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
