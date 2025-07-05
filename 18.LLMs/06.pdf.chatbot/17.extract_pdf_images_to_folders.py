import os
from pdf2image import convert_from_path

PDF_SOURCE_FOLDER = r'H:\Projects Control (PC)\10 Backup\06 Yasser\Damietta Buildings Project'
OUTPUT_ROOT = r'D:\OEssam\Test\demitta-pdf-images'
POPPLER_Path = r"C:\\Program Files\\poppler\\Library\\bin"

def safe_folder_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    # Remove or replace characters not allowed in Windows folder names
    return base.replace(':','_').replace('\\','_').replace('/','_').replace('*','_').replace('?','_').replace('"','_').replace('<','_').replace('>','_').replace('|','_')

def get_all_pdfs():
    pdfs = []
    for dirpath, _, filenames in os.walk(PDF_SOURCE_FOLDER):
        for fn in filenames:
            if fn.lower().endswith('.pdf'):
                full = os.path.join(dirpath, fn)
                pdfs.append(full)
    pdfs.sort()
    return pdfs

def extract_images(pdf_path, out_root):
    folder_name = safe_folder_name(pdf_path)
    out_dir = os.path.join(out_root, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_Path)
        for i, img in enumerate(images):
            out_name = f"page_{i+1}.png"
            out_path = os.path.join(out_dir, out_name)
            img.save(out_path, "PNG")
        print(f"Extracted {len(images)} images from {pdf_path} to {out_dir}")
    except Exception as e:
        print(f"[ERROR] Failed to extract images from {pdf_path}: {e}")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    pdfs = get_all_pdfs()
    if not pdfs:
        print("No PDFs found.")
        return
    for pdf_path in pdfs:
        extract_images(pdf_path, OUTPUT_ROOT)

if __name__ == "__main__":
    main()
