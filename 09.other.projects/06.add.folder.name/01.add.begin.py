import os
import tkinter as tk
from tkinter import filedialog
import sys
import re
import unicodedata

def add_parent_folder_name_to_files(root_folder):
    """
    Recursively go through all files in the root_folder and its subfolders
    and rename each file by adding the parent folder name followed by a dot to the filename.
    """
    # Count for statistics
    renamed_count = 0
    error_count = 0
    
    # Walk through all directories and files
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Skip the root folder itself
        if dirpath == root_folder:
            continue
            
        # Get the parent folder name
        parent_folder_name = os.path.basename(dirpath)
        
        # Clean the folder name to remove emojis and symbols
        parent_folder_name = clean_folder_name(parent_folder_name)
        
        # Process each file in the current directory
        for filename in filenames:
            # Skip files that already have the parent folder name
            if filename.startswith(f"{parent_folder_name}."):
                print(f"Skipping already renamed file: {filename}")
                continue
                
            # Create the new filename with parent folder name
            new_filename = f"{parent_folder_name}.{filename}"
            
            # Full paths for old and new files
            old_file_path = os.path.join(dirpath, filename)
            new_file_path = os.path.join(dirpath, new_filename)
            
            try:
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {filename}: {str(e)}")
                error_count += 1
    
    # Print summary
    print(f"\nRenaming complete!\nSuccessfully renamed: {renamed_count} files\nErrors: {error_count}")

def clean_folder_name(folder_name):
    """
    Remove emojis and symbols from the folder name, keeping only text characters.
    """
    # Remove emojis and other special characters
    # First, normalize to separate combined characters
    normalized = unicodedata.normalize('NFKD', folder_name)
    
    # Keep only letters, numbers, spaces, dots, underscores, and hyphens
    cleaned_name = re.sub(r'[^\w\s.-]', '', normalized, flags=re.ASCII)
    
    # Remove any leading/trailing whitespace
    cleaned_name = cleaned_name.strip()
    
    # If the cleaned name is empty, use a default name
    if not cleaned_name:
        cleaned_name = "folder"
    
    return cleaned_name


def browse_for_folder():
    """
    Open a folder browser dialog and return the selected folder path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Show the folder selection dialog
    folder_path = filedialog.askdirectory(title="Select the folder containing files to rename")
    
    # Close the tkinter window
    root.destroy()
    
    return folder_path

def main():
    print("File Renamer - Add Parent Folder Name")
    print("======================================")
    print("This script will rename all files by adding their parent folder name followed by a dot.")
    print("Example: folder/file.mp4 -> folder/foldername.file.mp4\n")
    
    # Get the folder path from command line or browse dialog
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        print("Please select the folder to process...")
        folder_path = browse_for_folder()
    
    # Check if a folder was selected
    if not folder_path:
        print("No folder selected. Exiting.")
        return
    
    print(f"Selected folder: {folder_path}")
    
    # Confirm before proceeding
    confirm = input(f"Do you want to rename all files in '{folder_path}' and its subfolders? (y/n): ").lower()
    
    if confirm == 'y':
        add_parent_folder_name_to_files(folder_path)
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    main()