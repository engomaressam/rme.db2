"""
Convert HTML batch visualizations to a single PDF using Selenium
"""
import os
import time
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Configuration
BATCH_DIR = "neo4j_exports"
OUTPUT_PDF = f"neo4j_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

# Set up Chrome options for headless mode
def get_chrome_options():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1920,1080')
    return chrome_options

def html_to_png(html_path, output_path):
    """Convert HTML to PNG using Selenium"""
    driver = None
    try:
        # Set up Chrome WebDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=get_chrome_options())
        
        # Load the HTML file
        driver.get(f"file://{os.path.abspath(html_path)}")
        
        # Wait for the visualization to load
        time.sleep(2)
        
        # Take screenshot
        driver.save_screenshot(output_path)
        return output_path
    except Exception as e:
        print(f"Error converting {html_path} to image: {e}")
        return None
    finally:
        if driver:
            driver.quit()

def create_pdf_from_images(image_files, output_pdf):
    """Combine images into a single PDF"""
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    
    for img_file in image_files:
        if img_file and os.path.exists(img_file):
            try:
                img = Image.open(img_file)
                img_width, img_height = img.size
                aspect = img_height / float(img_width)
                
                # Scale image to fit page width
                new_width = width - 100  # Add margins
                new_height = new_width * aspect
                
                # If image is too tall, scale to fit page height
                if new_height > height - 100:
                    new_height = height - 100
                    new_width = new_height / aspect
                
                # Center the image
                x = (width - new_width) / 2
                y = (height - new_height) / 2
                
                c.drawImage(img_file, x, y, width=new_width, height=new_height)
                c.showPage()
            except Exception as e:
                print(f"Error adding {img_file} to PDF: {e}")
    
    c.save()

def main():
    print("Starting HTML to PDF conversion...")
    
    # Find all batch HTML files
    batch_files = [f for f in os.listdir(BATCH_DIR) 
                  if f.startswith('batch_') and f.endswith('.html')]
    
    if not batch_files:
        print(f"No batch HTML files found in {BATCH_DIR}")
        return
    
    print(f"Found {len(batch_files)} batch files")
    
    # Create temporary directory for images
    temp_dir = os.path.join(BATCH_DIR, "temp_images")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Convert HTML files to images
    image_files = []
    for i, html_file in enumerate(sorted(batch_files)):
        print(f"Processing {html_file}...")
        html_path = os.path.join(BATCH_DIR, html_file)
        img_path = os.path.join(temp_dir, f"batch_{i+1}.png")
        
        # Try to convert HTML to image
        img_path = html_to_png(html_path, img_path)
        if img_path:
            image_files.append(img_path)
    
    if not image_files:
        print("No images were created. Please check if wkhtmltopdf is installed.")
        return
    
    # Create PDF
    output_path = os.path.join(BATCH_DIR, OUTPUT_PDF)
    create_pdf_from_images(image_files, output_path)
    
    print(f"\nPDF created successfully: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
