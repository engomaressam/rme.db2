import os
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image
import glob

def resize_gif(gif_path, min_dimension=400, quality_mode='high'):
    """
    Resize a GIF file to have a minimum dimension of min_dimension pixels
    while preserving the aspect ratio and maintaining color quality.
    
    Parameters:
    - gif_path: Path to the GIF file
    - min_dimension: Minimum dimension (width or height) in pixels
    - quality_mode: 'high' for better color preservation, 'normal' for standard resizing
    """
    try:
        # Open the GIF file
        img = Image.open(gif_path)
        width, height = img.size
        
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
            
            # Determine the resampling filter based on quality mode
            if quality_mode == 'high':
                resampling_filter = Image.LANCZOS  # Best quality, slower
            else:
                resampling_filter = Image.BILINEAR  # Good balance of quality and speed
            
            # Save all frames
            try:
                frame_count = 0
                while True:
                    # Copy the current frame
                    current_frame = img.copy()
                    
                    # For better color preservation in high quality mode
                    if quality_mode == 'high' and frame_count == 0:
                        # Convert to RGBA for better color handling
                        if current_frame.mode != 'RGBA':
                            current_frame = current_frame.convert('RGBA')
                    
                    # Resize the frame
                    resized_frame = current_frame.resize((new_width, new_height), resampling_filter)
                    
                    # Convert back to the original mode if needed
                    if quality_mode == 'high' and img.mode != 'RGBA' and resized_frame.mode == 'RGBA':
                        resized_frame = resized_frame.convert(img.mode)
                    
                    frames.append(resized_frame)
                    frame_count += 1
                    
                    # Move to the next frame
                    img.seek(img.tell() + 1)
            except EOFError:
                # End of frames
                pass
            
            # Save the resized GIF
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
                print(f"Resized animated GIF: {gif_path} to {new_width}x{new_height} with {quality_mode} quality")
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
                print(f"Resized static GIF: {gif_path} to {new_width}x{new_height} with {quality_mode} quality")
            
            return True
        else:
            print(f"No resize needed for: {gif_path} ({width}x{height})")
            return False
    except Exception as e:
        print(f"Error processing {gif_path}: {e}")
        return False

def process_folder(folder_path, quality_mode):
    """
    Recursively process all GIF files in the given folder and its subfolders.
    
    Parameters:
    - folder_path: Path to the folder containing GIFs
    - quality_mode: 'high' for better color preservation, 'normal' for standard resizing
    """
    # Count variables
    total_gifs = 0
    resized_gifs = 0
    
    # Find all GIF files recursively
    gif_files = glob.glob(os.path.join(folder_path, "**", "*.gif"), recursive=True)
    total_gifs = len(gif_files)
    
    print(f"Found {total_gifs} GIF files in {folder_path} and its subfolders.")
    
    # Process each GIF file
    for gif_path in gif_files:
        if resize_gif(gif_path, quality_mode=quality_mode):
            resized_gifs += 1
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total GIF files found: {total_gifs}")
    print(f"GIF files resized: {resized_gifs}")
    print(f"Quality mode used: {quality_mode}")

def main():
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()
    
    # Show folder selection dialog
    print("Please select a folder containing GIF files...")
    folder_path = filedialog.askdirectory(title="Select Folder Containing GIF Files")
    
    if folder_path:
        print(f"Selected folder: {folder_path}")
        
        # Ask for quality mode
        quality_choice = simpledialog.askstring(
            "Quality Settings", 
            "Select quality mode (type 'high' or 'normal'):\n\n" +
            "'high' - Better color preservation (recommended)\n" +
            "'normal' - Standard resizing",
            initialvalue="high"
        )
        
        # Default to high if canceled or invalid input
        if quality_choice not in ['high', 'normal']:
            quality_choice = 'high'
            print("Using default 'high' quality mode.")
        else:
            print(f"Using '{quality_choice}' quality mode.")
        
        process_folder(folder_path, quality_choice)
    else:
        print("No folder selected. Exiting.")
    
    # Close the hidden root window
    root.destroy()

if __name__ == "__main__":
    main()
