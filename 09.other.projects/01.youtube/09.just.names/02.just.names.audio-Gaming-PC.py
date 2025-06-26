import os
import tkinter as tk
from tkinter import filedialog

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.aac', '.flac', '.ogg', '.wma', '.m4a', '.alac', '.aiff', '.opus', '.amr', '.ape'}

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

def process_audios(root_dir):
    for folder, _, files in os.walk(root_dir):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext.lower() in AUDIO_EXTENSIONS:
                audio_path = os.path.join(folder, file)
                txt_path = os.path.join(folder, name + '.txt')
                try:
                    os.remove(audio_path)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(file)
                    print(f"Replaced: {audio_path} -> {txt_path}")
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")

def main():
    directory = get_directory()
    process_audios(directory)
    print('Done. All audio files replaced with text files containing their names.')

if __name__ == '__main__':
    main()
