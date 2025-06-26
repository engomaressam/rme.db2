import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from mutagen.flac import FLAC, Picture
from PIL import Image
import io

def tag_flac_files():
    # Create root window but hide it
    root = tk.Tk()
    root.withdraw()

    # Ask for the directory containing FLAC albums
    folder_path = filedialog.askdirectory(title="Select the folder containing FLAC albums")
    if not folder_path:
        print("No folder selected. Exiting...")
        return

    # Ask for the genre
    genre = simpledialog.askstring("Input", "Enter the genre for all albums:", parent=root)
    if not genre:
        print("No genre entered. Exiting...")
        return

    # Walk through all subdirectories
    for root_dir, dirs, files in os.walk(folder_path):
        # Filter for FLAC files
        flac_files = [f for f in files if f.lower().endswith('.flac')]
        
        if flac_files:
            # Get album name from the current directory name
            album_name = os.path.basename(root_dir)
            
            print(f"\nProcessing album: {album_name}")
            
            # Look for cover art
            cover_art = None
            cover_art_path = os.path.join(root_dir, "coverart.jpg")
            if os.path.exists(cover_art_path):
                try:
                    # Read and prepare the cover art
                    with Image.open(cover_art_path) as img:
                        # Convert to RGB if necessary (handles RGBA images)
                        if img.mode in ('RGBA', 'LA'):
                            img = img.convert('RGB')
                        
                        # Save to bytes
                        img_data = io.BytesIO()
                        img.save(img_data, format='JPEG')
                        img_data = img_data.getvalue()
                        
                        # Create FLAC picture
                        picture = Picture()
                        picture.type = 3  # Front cover
                        picture.mime = "image/jpeg"
                        picture.desc = "Front cover"
                        picture.data = img_data
                        
                        cover_art = picture
                        print(f"Found cover art: {cover_art_path}")
                except Exception as e:
                    print(f"Error processing cover art: {str(e)}")
            
            # Process each FLAC file in the current directory
            for flac_file in flac_files:
                file_path = os.path.join(root_dir, flac_file)
                try:
                    audio = FLAC(file_path)
                    
                    # Set the album and genre tags
                    audio['album'] = album_name
                    audio['genre'] = genre
                    
                    # Clear existing pictures and add the new cover art if available
                    if cover_art:
                        audio.clear_pictures()
                        audio.add_picture(cover_art)
                    
                    # Save the changes
                    audio.save()
                    print(f"Tagged: {flac_file}")
                    
                except Exception as e:
                    print(f"Error processing {flac_file}: {str(e)}")

    print("\nTagging complete!")

if __name__ == "__main__":
    tag_flac_files() 