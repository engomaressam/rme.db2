import os
import multiprocessing
from pdf2image import convert_from_path

PDF_SOURCE_FOLDER = r'D:\OEssam\Test\EIPICO 3'
OUTPUT_ROOT = r'D:\OEssam\Test\epico-pdf-images'
POPPLER_Path = r"C:\\Program Files\\poppler\\Library\\bin"
PAGE_LIMIT = 10
TIMEOUT_SECONDS = 60

def safe_folder_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
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

def extract_images_worker(pdf_path, out_dir, result_queue):
    try:
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_Path, first_page=1, last_page=PAGE_LIMIT)
        for i, img in enumerate(images):
            out_name = f"page_{i+1}.png"
            out_path = os.path.join(out_dir, out_name)
            img.save(out_path, "PNG")
        result_queue.put((len(images), None))
    except Exception as e:
        result_queue.put((0, str(e)))

def extract_images(pdf_path, out_root):
    folder_name = safe_folder_name(pdf_path)
    out_dir = os.path.join(out_root, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=extract_images_worker, args=(pdf_path, out_dir, result_queue))
    proc.start()
    proc.join(TIMEOUT_SECONDS)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"[TIMEOUT] Skipped {pdf_path} (exceeded {TIMEOUT_SECONDS}s)")
        return
    if not result_queue.empty():
        num_images, error = result_queue.get()
        if error:
            print(f"[ERROR] Failed to extract images from {pdf_path}: {error}")
        else:
            print(f"Extracted {num_images} images from {pdf_path} to {out_dir}")
    else:
        print(f"[ERROR] Unknown failure for {pdf_path}")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    pdfs = get_all_pdfs()
    if not pdfs:
        print("No PDFs found.")
        return
    for pdf_path in pdfs:
        extract_images(pdf_path, OUTPUT_ROOT)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
