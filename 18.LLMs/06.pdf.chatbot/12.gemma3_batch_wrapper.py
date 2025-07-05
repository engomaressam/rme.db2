import os
import subprocess
import time

# Path to the working script
SCRIPT_PATH = r"c:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\18.llms\06.start.over\01.fill.extra\09.gemma3_full_pdf_processing.py"

# Output directory and processed files log
OUTPUT_DIR = r'D:\OEssam\Test\gemma3'
PROCESSED_FILES_LOG = os.path.join(OUTPUT_DIR, 'processed_pdfs.txt')

def get_processed_count():
    """Get the number of PDFs that have been processed so far"""
    if not os.path.exists(PROCESSED_FILES_LOG):
        return 0
    
    with open(PROCESSED_FILES_LOG, 'r', encoding='utf-8') as f:
        return len([line for line in f if line.strip()])

def main():
    """Run the 09.gemma3_full_pdf_processing.py script repeatedly until all PDFs are processed"""
    print("=== Starting batch processing using the working script ===")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Keep track of how many PDFs have been processed
    last_processed_count = get_processed_count()
    print(f"Starting with {last_processed_count} already processed PDFs")
    
    # Run the script repeatedly
    while True:
        print("\n" + "="*50)
        print(f"Running script to process the next PDF...")
        print("="*50 + "\n")
        
        # Run the script
        process = subprocess.Popen(
            ["python", SCRIPT_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream the output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Wait for the process to complete
        process.wait()
        
        # Check if any new PDFs were processed
        current_processed_count = get_processed_count()
        if current_processed_count > last_processed_count:
            print(f"\nSuccessfully processed a new PDF. Total processed: {current_processed_count}")
            last_processed_count = current_processed_count
        else:
            print("\nNo new PDFs were processed. All PDFs must be completed or there was an error.")
            print("Batch processing complete.")
            break
        
        # Small delay before starting the next run
        time.sleep(2)

if __name__ == "__main__":
    main()
