import os
import shutil
import re
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

def organize_images_by_prefix(folder_path):
    """
    Organizes JPG files in the given folder into subfolders based on the first two digits of their names.
    
    Args:
        folder_path (str): Path to the folder containing JPG files
    """
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return
    
    # Get all JPG files in the folder
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg') and f.startswith('-')]
    
    if not jpg_files:
        print(f"No JPG files found in {folder_path}")
        return
    
    print(f"Found {len(jpg_files)} JPG files to organize.")
    
    # Process each file
    for file_name in jpg_files:
        # Extract the first two digits after the dash
        match = re.match(r'-([0-9]{2}).*', file_name)
        if match:
            prefix = match.group(1)  # Get the first two digits
            
            # Create subfolder if it doesn't exist
            subfolder_path = os.path.join(folder_path, prefix)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                print(f"Created subfolder: {prefix}")
            
            # Move the file to the subfolder
            source_path = os.path.join(folder_path, file_name)
            dest_path = os.path.join(subfolder_path, file_name)
            
            try:
                shutil.move(source_path, dest_path)
                print(f"Moved {file_name} to folder {prefix}")
            except Exception as e:
                print(f"Error moving {file_name}: {e}")
        else:
            print(f"Skipping {file_name} - doesn't match expected pattern")

def browse_folder():
    """
    Opens a folder browser dialog and returns the selected folder path.
    """
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()
    
    # Show the folder selection dialog
    folder_path = filedialog.askdirectory(title="Select Folder Containing JPG Files")
    
    # Destroy the root window
    root.destroy()
    
    return folder_path

# Main execution
if __name__ == "__main__":
    print("JPG File Organizer")
    print("-" * 50)
    
    # Get folder path from command line arguments, GUI dialog, or user input
    folder_path = ""
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        print(f"Using folder path from command line: {folder_path}")
    else:
        # Try to use the GUI folder browser
        try:
            folder_path = browse_folder()
            if folder_path:  # If a folder was selected
                print(f"Selected folder: {folder_path}")
            else:  # If dialog was canceled
                print("Folder selection canceled.")
                sys.exit(0)
        except Exception as e:
            print(f"Error with folder browser: {e}")
            # Fall back to console input
            try:
                folder_path = input("Enter the path to the folder containing JPG files: ").strip()
            except EOFError:
                print("Error: This script requires a folder path.")
                print("Usage: python script.py [folder_path]")
                sys.exit(1)
    
    # Remove quotes if the user included them
    folder_path = folder_path.strip('"\'')
    
    if not folder_path:
        print("No folder path provided. Exiting.")
        sys.exit(1)
    
    # Organize the files
    organize_images_by_prefix(folder_path)
    
    print("\nOrganization complete!")
    print("Files have been moved to numbered subfolders based on their first two digits.")
    
    # Show completion message in GUI if we used the GUI
    if len(sys.argv) <= 1:
        try:
            # Create a root window but hide it
            root = tk.Tk()
            root.withdraw()
            
            # Show message box
            messagebox.showinfo("Complete", "Files have been organized into numbered subfolders based on their first two digits.")
            
            # Destroy the root window
            root.destroy()
        except Exception as e:
            print(f"Error showing completion message: {e}")
            # Fall back to console
            try:
                input("Press Enter to exit...")
            except EOFError:
                pass