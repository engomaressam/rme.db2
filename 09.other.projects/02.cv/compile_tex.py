import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

def compile_latex(tex_file):
    """Compile a LaTeX file using pdflatex."""
    # Get the directory of the tex file
    tex_dir = os.path.dirname(os.path.abspath(tex_file))
    tex_filename = os.path.basename(tex_file)
    
    # Change to the tex file's directory
    os.chdir(tex_dir)
    print(f"Working directory: {tex_dir}")
    print(f"Compiling {tex_filename}...")
    
    # Run pdflatex
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_filename],
            capture_output=True,
            text=True
        )
        
        # Check if PDF was created
        pdf_file = tex_filename.replace('.tex', '.pdf')
        if os.path.exists(pdf_file):
            pdf_path = os.path.abspath(pdf_file)
            print(f"\nSuccess! PDF created at: {pdf_path}")
            print(f"PDF size: {os.path.getsize(pdf_file) / 1024:.1f} KB")
            
            # Try to open the PDF
            try:
                os.startfile(pdf_path)
                print("Opening PDF...")
            except:
                print("Note: Could not automatically open the PDF.")
        else:
            print("\nError: PDF file was not created")
            print("\nCompiler output:")
            print(result.stdout)
            print("\nErrors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error running pdflatex: {e}")

def browse_tex_file():
    """Open a file browser dialog to select a .tex file."""
    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()
    
    # Open the file dialog
    file_path = filedialog.askopenfilename(
        title="Select LaTeX file to compile",
        filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")],
        initialdir=os.getcwd()
    )
    
    return file_path if file_path else None

if __name__ == "__main__":
    print("LaTeX Compiler")
    print("-------------")
    
    # Get the .tex file path
    if len(sys.argv) > 1:
        tex_file = sys.argv[1]
    else:
        tex_file = browse_tex_file()
    
    if not tex_file:
        print("No file selected. Exiting...")
        sys.exit(1)
    
    if not os.path.exists(tex_file):
        print(f"Error: File '{tex_file}' not found!")
        sys.exit(1)
    
    compile_latex(tex_file) 