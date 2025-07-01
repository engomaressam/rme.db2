"""
Execute all notebooks and save their outputs to text files
"""
import os
import subprocess
from datetime import datetime

# Create output directory
output_dir = "notebook_outputs"
os.makedirs(output_dir, exist_ok=True)

# List of notebooks to run
notebooks = [
    "01.cross_entropy_logistic_regression_v2.ipynb",
    "02.softmax_in_one_dimension_v2.ipynb",
    "03.lab_predicting _MNIST_using_Softmax_v2.ipynb"
]

for nb in notebooks:
    if not os.path.exists(nb):
        print(f"Notebook not found: {nb}")
        continue
        
    print(f"\nExecuting {nb}...")
    output_file = os.path.join(output_dir, f"{os.path.splitext(nb)[0]}_output.txt")
    
    try:
        # Run the notebook and capture output
        result = subprocess.run(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--stdout", nb],
            capture_output=True,
            text=True
        )
        
        # Save output to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {nb} ===\n")
            f.write(f"Executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== Output ===\n")
            f.write(result.stdout)
            
            if result.stderr:
                f.write("\n=== Errors ===\n")
                f.write(result.stderr)
        
        print(f"  Output saved to: {output_file}")
        
    except Exception as e:
        print(f"  Error executing {nb}: {str(e)}")

print("\nAll notebooks executed.")

# Add GPU info at the end
try:
    import torch
    gpu_file = os.path.join(output_dir, "gpu_info.txt")
    with open(gpu_file, 'w') as f:
        f.write("=== GPU Information ===\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    print(f"\nGPU info saved to: {gpu_file}")
except Exception as e:
    print(f"\nCould not save GPU info: {str(e)}")
