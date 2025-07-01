"""
Run a single notebook and save output to a text file
"""
import os
import sys
import subprocess
from datetime import datetime

def run_notebook(notebook_path):
    """Run a notebook and save output to a text file"""
    if not os.path.exists(notebook_path):
        print(f"Error: {notebook_path} not found")
        return
    
    print(f"Running {notebook_path}...")
    
    # Create output directory
    output_dir = "notebook_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(notebook_path))[0]
    output_file = os.path.join(output_dir, f"{base_name}_output.txt")
    
    try:
        # Run the notebook and capture output
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--stdout', notebook_path],
            capture_output=True, 
            text=True
        )
        
        # Save output to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Output of {notebook_path} ===\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout + "\n\n")
            if result.stderr:
                f.write("=== STDERR ===\n")
                f.write(result.stderr + "\n")
        
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error running {notebook_path}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_single_notebook.py <notebook_path>")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    run_notebook(notebook_path)
