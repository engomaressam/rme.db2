import os
import tkinter as tk
from tkinter import filedialog
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def convert_flac_to_mp3(flac_path, mp3_path, album_name):
    """Convert a single FLAC file to MP3 using ffmpeg."""
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
        
        # First verify the FLAC file exists and has size > 0
        if not os.path.exists(flac_path):
            print(f"Error: Source file does not exist: {flac_path}")
            return False, album_name
            
        if os.path.getsize(flac_path) == 0:
            print(f"Error: Source file is empty: {flac_path}")
            return False, album_name
        
        # Use ffmpeg to convert FLAC to MP3 while copying metadata
        cmd = [
            'ffmpeg', '-i', flac_path,
            '-ab', '320k',
            '-map_metadata', '0',
            '-id3v2_version', '3',
            '-y',  # Overwrite output file if it exists
            mp3_path
        ]
        
        # Run ffmpeg
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check the output file was created and has size > 0
        if process.returncode == 0 and os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
            print(f"Converted: {os.path.basename(flac_path)}")
            return True, album_name
        else:
            # If conversion failed or output is empty, show the ffmpeg output
            print(f"\nError converting {os.path.basename(flac_path)}:")
            print("ffmpeg output:")
            print(process.stderr)
            # Clean up empty or partial files
            if os.path.exists(mp3_path):
                os.remove(mp3_path)
            return False, album_name
            
    except Exception as e:
        print(f"Error converting {os.path.basename(flac_path)}: {str(e)}")
        # Clean up empty or partial files
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        return False, album_name

def copy_cover_art(src_dir, dst_dir):
    """Copy coverart.jpg from source to destination directory if it exists."""
    cover_art = os.path.join(src_dir, "coverart.jpg")
    if os.path.exists(cover_art):
        try:
            shutil.copy2(cover_art, os.path.join(dst_dir, "coverart.jpg"))
            print(f"Copied cover art to: {dst_dir}")
        except Exception as e:
            print(f"Error copying cover art to {dst_dir}: {str(e)}")

def print_album_separator(album_name):
    """Print a separator with album name."""
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"Processing Album: {album_name}")
    print(f"{separator}")

def convert_library():
    # Create root window but hide it
    root = tk.Tk()
    root.withdraw()

    # Ask for the source FLAC directory
    flac_dir = filedialog.askdirectory(title="Select the source FLAC directory")
    if not flac_dir:
        print("No source directory selected. Exiting...")
        return

    # Ask for the destination MP3 directory
    mp3_dir = filedialog.askdirectory(title="Select the destination MP3 directory")
    if not mp3_dir:
        print("No destination directory selected. Exiting...")
        return

    # Create the destination directory if it doesn't exist
    os.makedirs(mp3_dir, exist_ok=True)

    # Dictionary to store conversion tasks by album
    conversion_tasks = {}
    error_files = []
    current_album = None

    # Walk through all subdirectories
    for root_dir, dirs, files in os.walk(flac_dir):
        # Filter for FLAC files
        flac_files = [f for f in files if f.lower().endswith('.flac')]
        
        if flac_files:
            # Get the relative path from the FLAC root directory
            rel_path = os.path.relpath(root_dir, flac_dir)
            # Create the corresponding MP3 directory path
            mp3_album_dir = os.path.join(mp3_dir, rel_path)
            
            # Create the MP3 album directory
            os.makedirs(mp3_album_dir, exist_ok=True)
            
            # Get album name from the folder name
            album_name = os.path.basename(root_dir)
            
            # Print album separator
            print_album_separator(album_name)
            
            # Copy cover art if it exists
            copy_cover_art(root_dir, mp3_album_dir)
            
            # Process each FLAC file in the current directory
            album_tasks = []
            for flac_file in flac_files:
                flac_path = os.path.join(root_dir, flac_file)
                mp3_file = os.path.splitext(flac_file)[0] + '.mp3'
                mp3_path = os.path.join(mp3_album_dir, mp3_file)
                
                # Add the conversion task to our list
                album_tasks.append((flac_path, mp3_path, album_name))
            
            conversion_tasks[album_name] = album_tasks

    # Convert files using a thread pool
    print("\nStarting conversion process...")
    with ThreadPoolExecutor() as executor:
        # Process each album's files
        for album_name, tasks in conversion_tasks.items():
            # Submit all conversion tasks for this album
            futures = [
                executor.submit(convert_flac_to_mp3, flac_path, mp3_path, album)
                for flac_path, mp3_path, album in tasks
            ]
            
            # Wait for all tasks in this album to complete
            for future, (flac_path, _, _) in zip(futures, tasks):
                success, album = future.result()
                if not success:
                    error_files.append((flac_path, album))

    if error_files:
        print("\nThe following files had conversion errors:")
        current_album = None
        for file, album in error_files:
            if album != current_album:
                print(f"\nIn album {album}:")
                current_album = album
            print(f"- {os.path.basename(file)}")
    
    print("\nConversion complete!")

if __name__ == "__main__":
    convert_library() 