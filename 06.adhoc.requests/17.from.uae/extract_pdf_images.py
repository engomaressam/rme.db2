import fitz  # PyMuPDF
import os

# Define input PDF and output directory
input_pdf_path = r"d:\OneDrive2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\06.adhoc.requests\17.from.uae\ADWEA_APPROVED_CONTRACTORS_LIST.pdf"
output_dir = r"d:\OneDrive2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\06.adhoc.requests\17.from.uae"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Page numbers to extract images from (1-indexed)
pages_to_extract = [3, 13, 23]

print(f"Opening PDF: {input_pdf_path}")

try:
    doc = fitz.open(input_pdf_path)
except Exception as e:
    print(f"Error opening PDF: {e}")
    exit()

for page_num_1_indexed in pages_to_extract:
    page_num_0_indexed = page_num_1_indexed - 1  # PyMuPDF uses 0-indexed pages
    if page_num_0_indexed < 0 or page_num_0_indexed >= len(doc):
        print(f"Page number {page_num_1_indexed} is out of range. PDF has {len(doc)} pages.")
        continue

    page = doc.load_page(page_num_0_indexed)
    image_list = page.get_images(full=True)

    if not image_list:
        print(f"No images found on page {page_num_1_indexed}.")
        continue

    print(f"Found {len(image_list)} images on page {page_num_1_indexed}.")

    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        # Save as JPG, if not already JPG, Pillow might be needed for conversion
        # For simplicity, we'll save with the original extension if it's common, or try JPG
        output_image_path = os.path.join(output_dir, f"page_{page_num_1_indexed}_image_{img_index + 1}.{image_ext if image_ext in ['jpeg', 'jpg', 'png'] else 'jpg'}")
        
        try:
            with open(output_image_path, "wb") as img_file:
                img_file.write(image_bytes)
            print(f"Saved image: {output_image_path}")
        except Exception as e:
            print(f"Error saving image {output_image_path}: {e}")

doc.close()
print("Image extraction complete.")
