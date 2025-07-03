import os
from docx2pdf import convert

# Directory containing the .docx files
src_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(src_dir, "pdfs")
os.makedirs(pdf_dir, exist_ok=True)

# List all .docx files (excluding temporary/system files)
docx_files = [f for f in os.listdir(src_dir) if f.endswith(".docx") and not f.startswith("~$")]

for docx_file in docx_files:
    src_path = os.path.join(src_dir, docx_file)
    pdf_path = os.path.join(pdf_dir, docx_file.replace(".docx", ".pdf"))
    convert(src_path, pdf_path)
    print(f"Converted {docx_file} -> {pdf_path}")

print("All conversions complete.") 