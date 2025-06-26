import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageFile
import glob
import shutil

# Allow truncated images to be processed
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_gif(gif_path, min_dimension=400):
    """
    Resize a GIF file to have a minimum dimension of min_dimension pixels
    while preserving the aspect ratio and maintaining high color quality.
    
    Parameters:
    - gif_path: Path to the GIF file
    - min_dimension: Minimum dimension (width or height) in pixels
    """
    try:
        # First try to verify the file is a valid GIF
        try:
            # Open the GIF file
            img = Image.open(gif_path)
            # Try to access the size to verify it can be read
            width, height = img.size
        except Exception as verify_error:
            print(f"Warning: Corrupted GIF detected: {gif_path}")
            print(f"Error details: {verify_error}")
            
            # Create a backup of the original file
            backup_path = gif_path + ".backup"
            try:
                shutil.copy2(gif_path, backup_path)
                print(f"Created backup at: {backup_path}")
            except Exception as backup_error:
                print(f"Failed to create backup: {backup_error}")
                return False
            
            # Try to repair by opening with LOAD_TRUNCATED_IMAGES enabled
            try:
                img = Image.open(gif_path)
                width, height = img.size
                print(f"Successfully repaired truncated GIF: {gif_path}")
            except Exception as repair_error:
                print(f"Failed to repair GIF: {repair_error}")
                # Restore from backup if repair failed
                try:
                    shutil.copy2(backup_path, gif_path)
                    print(f"Restored original file from backup")
                except Exception as restore_error:
                    print(f"Failed to restore from backup: {restore_error}")
                return False
        
        # Check if resizing is needed
        if width < min_dimension or height < min_dimension:
            # Calculate new dimensions while preserving aspect ratio
            if width < height:
                # Width is the smaller dimension
                new_width = min_dimension
                new_height = int(height * (min_dimension / width))
            else:
                # Height is the smaller dimension
                new_height = min_dimension
                new_width = int(width * (min_dimension / height))
            
            # Create a list to store all frames
            frames = []
            
            # Get the original palette and transparency info
            original_palette = None
            if 'palette' in img.info:
                original_palette = img.info['palette']
            
            # Use LANCZOS resampling filter for best quality
            resampling_filter = Image.LANCZOS
            
            # Save all frames
            try:
                frame_count = 0
                while True:
                    try:
                        # Copy the current frame
                        current_frame = img.copy()
                        
                        # For better color preservation
                        if frame_count == 0:
                            # Convert to RGBA for better color handling
                            if current_frame.mode != 'RGBA':
                                current_frame = current_frame.convert('RGBA')
                        
                        # Resize the frame
                        resized_frame = current_frame.resize((new_width, new_height), resampling_filter)
                        
                        # Convert back to the original mode if needed
                        if img.mode != 'RGBA' and resized_frame.mode == 'RGBA':
                            resized_frame = resized_frame.convert(img.mode)
                        
                        frames.append(resized_frame)
                        frame_count += 1
                        
                        # Move to the next frame
                        img.seek(img.tell() + 1)
                    except Exception as frame_error:
                        print(f"Warning: Error processing frame {frame_count}: {frame_error}")
                        # Try to continue with the frames we have
                        break
            except EOFError:
                # End of frames - this is normal
                pass
            
            # If we couldn't extract any frames, return failure
            if len(frames) == 0:
                print(f"Error: Could not extract any frames from {gif_path}")
                return False
            
            # Save the resized GIF
            try:
                if len(frames) > 1:
                    # Save as animated GIF
                    save_kwargs = {
                        'format': 'GIF',
                        'append_images': frames[1:],
                        'save_all': True,
                        'duration': img.info.get('duration', 100),
                        'loop': img.info.get('loop', 0),
                        'optimize': False  # Don't optimize to preserve colors
                    }
                    
                    # Add transparency if present in original
                    if 'transparency' in img.info:
                        save_kwargs['transparency'] = img.info['transparency']
                    
                    # Add original palette if available
                    if original_palette:
                        save_kwargs['palette'] = original_palette
                    
                    frames[0].save(gif_path, **save_kwargs)
                    print(f"Resized animated GIF: {gif_path} to {new_width}x{new_height}")
                else:
                    # Save as static GIF
                    save_kwargs = {
                        'format': 'GIF',
                        'optimize': False  # Don't optimize to preserve colors
                    }
                    
                    # Add transparency if present in original
                    if 'transparency' in img.info:
                        save_kwargs['transparency'] = img.info['transparency']
                    
                    # Add original palette if available
                    if original_palette:
                        save_kwargs['palette'] = original_palette
                    
                    frames[0].save(gif_path, **save_kwargs)
                    print(f"Resized static GIF: {gif_path} to {new_width}x{new_height}")
                
                return True
            except Exception as save_error:
                print(f"Error saving resized GIF: {save_error}")
                # Try to restore from backup if we created one
                backup_path = gif_path + ".backup"
                if os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, gif_path)
                        print(f"Restored original file from backup after save failure")
                    except Exception as restore_error:
                        print(f"Failed to restore from backup: {restore_error}")
                return False
        else:
            print(f"No resize needed for: {gif_path} ({width}x{height})")
            return False
    except Exception as e:
        print(f"Error processing {gif_path}: {e}")
        print(f"File size: {os.path.getsize(gif_path)} bytes")
        return False

def process_folder(folder_path):
    """
    Recursively process all GIF files in the given folder and its subfolders.
    Always uses high quality mode.
    
    Parameters:
    - folder_path: Path to the folder containing GIFs
    """
    # Count variables
    total_gifs = 0
    resized_gifs = 0
    skipped_gifs = 0
    
    # Find all GIF files recursively
    gif_files = glob.glob(os.path.join(folder_path, "**", "*.gif"), recursive=True)
    total_gifs = len(gif_files)
    
    print(f"Found {total_gifs} GIF files in {folder_path} and its subfolders.")
    print("Using high quality mode for all GIFs.")
    
    # Process each GIF file
    for gif_path in gif_files:
        if resize_gif(gif_path):
            resized_gifs += 1
        else:
            skipped_gifs += 1
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total GIF files found: {total_gifs}")
    print(f"GIF files resized: {resized_gifs}")
    print(f"GIF files skipped: {skipped_gifs}")
    print(f"Quality mode used: high (hardcoded)")

def main():
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()
    
    # Show folder selection dialog
    print("Please select a folder containing GIF files...")
    folder_path = filedialog.askdirectory(title="Select Folder Containing GIF Files")
    
    if folder_path:
        print(f"Selected folder: {folder_path}")
        process_folder(folder_path)
    else:
        print("No folder selected. Exiting.")
    
    # Close the hidden root window
    root.destroy()

if __name__ == "__main__":
    main()
