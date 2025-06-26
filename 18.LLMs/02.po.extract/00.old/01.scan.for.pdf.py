import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox

def scan_pdfs(folder_path):
    """
    Scan a folder and all its subfolders for PDF files and return the count and list of filenames.
    
    Args:
        folder_path (str): Path to the folder to scan
        
    Returns:
        tuple: (count of PDFs, list of PDF filenames with relative paths)
    """
    # Ensure the path exists
    if not os.path.exists(folder_path):
        messagebox.showerror("Error", f"The path '{folder_path}' does not exist.")
        return 0, []
    
    # Ensure the path is a directory
    if not os.path.isdir(folder_path):
        messagebox.showerror("Error", f"'{folder_path}' is not a directory.")
        return 0, []
    
    pdf_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        # Find all PDF files in the current directory
        for file in files:
            if file.lower().endswith('.pdf'):
                # Get the full path to the PDF file
                full_path = os.path.join(root, file)
                # Store the full path and the relative path from the base folder
                rel_path = os.path.relpath(full_path, folder_path)
                pdf_files.append((full_path, rel_path))
    
    return len(pdf_files), pdf_files

def browse_folder():
    """
    Open a file dialog to browse for a folder and return the selected path.
    
    Returns:
        str: Selected folder path or empty string if canceled
    """
    folder_path = filedialog.askdirectory(title="Select Folder Containing PDF Files")
    return folder_path

def main():
    # Create root window
    root = tk.Tk()
    root.title("PDF Scanner")
    root.geometry("600x400")
    
    # Hide the main window initially
    root.withdraw()
    
    # Open folder browser dialog
    folder_path = browse_folder()
    
    if not folder_path:
        messagebox.showinfo("Canceled", "Folder selection was canceled.")
        root.destroy()
        return
    
    # Scan for PDFs
    count, filenames = scan_pdfs(folder_path)
    
    # Show the main window
    root.deiconify()
    
    # Create frame for results
    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Display results
    if count == 0:
        label = tk.Label(frame, text="No PDF files found in the selected directory.", font=("Arial", 12))
        label.pack(pady=10)
    else:
        header = tk.Label(frame, text=f"Found {count} PDF file(s) in:\n{folder_path}", font=("Arial", 12, "bold"))
        header.pack(pady=10)
        
        # Create a scrollable text area for file list
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        file_list = tk.Text(frame, height=15, width=70, yscrollcommand=scrollbar.set)
        file_list.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar.config(command=file_list.yview)
        
        # Add files to the list
        for i, (full_path, rel_path) in enumerate(filenames, 1):
            file_list.insert(tk.END, f"{i}. {rel_path}\n")
        
        file_list.config(state=tk.DISABLED)  # Make read-only
    
    # Add a close button
    close_button = tk.Button(frame, text="Close", command=root.destroy, width=10)
    close_button.pack(pady=10)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
