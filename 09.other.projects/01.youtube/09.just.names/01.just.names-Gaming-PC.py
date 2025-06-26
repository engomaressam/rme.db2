import os
import tkinter as tk
from tkinter import filedialog

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.3gp', '.m4v', '.ts', '.vob', '.ogv'}

def get_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring the dialog to the front
    path = filedialog.askdirectory(title='Select a directory to process')
    root.destroy()
    if not path:
        print('No directory selected. Exiting.')
        exit(0)
    return path

def process_videos(root_dir):
    for folder, _, files in os.walk(root_dir):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in VIDEO_EXTENSIONS:
                video_path = os.path.join(folder, file)
                txt_path = os.path.join(folder, name + '.txt')
                try:
                    os.remove(video_path)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(file)
                    print(f"Replaced: {video_path} -> {txt_path}")
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")

def main():
    directory = get_directory()
    process_videos(directory)
    print('Done. All videos replaced with text files containing their names.')

if __name__ == '__main__':
    main()
