"""
Run all notebooks from the notebooks/ directory and save outputs with GPU monitoring
"""
import os
import json
import subprocess
from datetime import datetime

# Configuration
NOTEBOOKS_DIR = "notebooks"
OUTPUT_DIR = "notebook_outputs"

def get_gpu_info():
    """Get GPU information"""
    try:
        import torch
        info = ["=== GPU Information ==="]
        info.append(f"PyTorch Version: {torch.__version__}")
        info.append(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            info.append(f"CUDA Version: {torch.version.cuda}")
            info.append(f"GPU: {torch.cuda.get_device_name(0)}")
            info.append(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            info.append(f"Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        
        return "\n".join(info)
    except Exception as e:
        return f"Could not get GPU info: {str(e)}"

def run_notebook(notebook_path, output_dir):
    """Run a notebook and save its output"""
    print(f"\nProcessing {os.path.basename(notebook_path)}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(
        output_dir, 
        f"{os.path.splitext(os.path.basename(notebook_path))[0]}_output.txt"
    )
    
    try:
        # Run the notebook and capture output
        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--stdout", notebook_path],
            capture_output=True,
            text=True
        )
        
        # Save output to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {os.path.basename(notebook_path)} ===\n")
            f.write(f"Executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(get_gpu_info() + "\n\n")
            f.write("=== Output ===\n")
            f.write(result.stdout)
            
            if result.stderr:
                f.write("\n=== Errors ===\n")
                f.write(result.stderr)
        
        print(f"  Output saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"  Error executing notebook: {str(e)}")
        return False

def main():
    # Get list of all notebook files
    notebook_files = [f for f in os.listdir(NOTEBOOKS_DIR) 
                     if f.endswith('.ipynb') and not f.startswith('.')]
    
    if not notebook_files:
        print(f"No notebook files found in {NOTEBOOKS_DIR}")
        return
    
    print(f"Found {len(notebook_files)} notebooks to process...")
    
    # Process each notebook
    success_count = 0
    for nb_file in sorted(notebook_files):
        nb_path = os.path.join(NOTEBOOKS_DIR, nb_file)
        if run_notebook(nb_path, OUTPUT_DIR):
            success_count += 1
    
    print(f"\nProcessing complete. Successfully processed {success_count} of {len(notebook_files)} notebooks.")
    print(f"Outputs saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
