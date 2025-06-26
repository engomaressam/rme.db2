import os
from PyPDF2 import PdfReader
from fpdf import FPDF

# List of malformed PDFs to fix
malformed_pdfs = [
    "03.warehouse/01.mcp_erp/table_docs/AP_AP_INVOICE_PAYMENTS_ALL_documentation.pdf",
    "03.warehouse/01.mcp_erp/table_docs/AP_AP_SUPPLIER_SITES_ALL_documentation.pdf",
    "03.warehouse/01.mcp_erp/table_docs/AP_AP_SUPPLIERS_documentation.pdf",
    "03.warehouse/01.mcp_erp/table_docs/AP_AP_INVOICES_ALL_documentation.pdf"
]

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def write_text_to_pdf(text, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.splitlines():
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

for pdf_path in malformed_pdfs:
    print(f"Processing {pdf_path} ...")
    text = extract_text_from_pdf(pdf_path)
    fixed_path = pdf_path.replace(".pdf", "_fixed.pdf")
    write_text_to_pdf(text, fixed_path)
    print(f"Fixed PDF saved to {fixed_path}") 