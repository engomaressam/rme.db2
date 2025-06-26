"""
Combine multiple PNG images into a single PDF file.
"""
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

# Configuration
IMAGE_DIR = "neo4j_exports/temp_images"
OUTPUT_PDF = f"neo4j_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

def create_pdf(image_dir, output_pdf):
    """Create a PDF from all PNG images in the directory."""
    # Get all PNG files in the directory
    image_files = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    
    if not image_files:
        print("No PNG files found in the directory.")
        return
    
    # Create a new PDF with Reportlab
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter  # Default to letter size (8.5 x 11 inches)
    
    print(f"Found {len(image_files)} images. Creating PDF...")
    
    for i, img_path in enumerate(image_files, 1):
        try:
            # Open the image file
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Calculate aspect ratio
            aspect = img_height / float(img_width)
            
            # Scale image to fit page width (with margins)
            margin = 50  # 0.5 inch margin
            max_width = width - 2 * margin
            max_height = height - 2 * margin
            
            new_width = min(max_width, img_width)
            new_height = new_width * aspect
            
            # If the image is too tall, scale it down to fit the page height
            if new_height > max_height:
                new_height = max_height
                new_width = new_height / aspect
            
            # Center the image on the page
            x = (width - new_width) / 2
            y = (height - new_height) / 2
            
            # Add the image to the PDF
            c.drawImage(img_path, x, y, width=new_width, height=new_height)
            c.showPage()  # Add a new page for the next image
            
            print(f"Added image {i}/{len(image_files)}: {os.path.basename(img_path)}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save the PDF
    c.save()
    print(f"\nPDF created successfully: {os.path.abspath(output_pdf)}")

if __name__ == "__main__":
    create_pdf(IMAGE_DIR, OUTPUT_PDF)
