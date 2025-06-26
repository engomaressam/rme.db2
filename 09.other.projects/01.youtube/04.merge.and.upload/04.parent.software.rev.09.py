#!/usr/bin/env python3
import os
import glob
import json
import time
import shutil
import logging
import re
import sys
from datetime import datetime
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import ssl
import httplib2
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('video_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

class ProgressWindow:
    def __init__(self, title="Progress", parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title(title)
        self.root.geometry("600x200")
        
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
        # Make window stay on top
        self.root.attributes('-topmost', True)
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var,
            maximum=100,
            length=500
        )
        self.progress_bar.pack(pady=5)
        
        # Details frame
        details_frame = ttk.LabelFrame(main_frame, text="Details", padding="5")
        details_frame.pack(fill=tk.X, pady=5)
        
        # Time info
        time_frame = ttk.Frame(details_frame)
        time_frame.pack(fill=tk.X, pady=2)
        
        self.elapsed_var = tk.StringVar(value="Elapsed: 0:00:00")
        self.remaining_var = tk.StringVar(value="Remaining: --:--:--")
        
        ttk.Label(time_frame, textvariable=self.elapsed_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(time_frame, textvariable=self.remaining_var).pack(side=tk.RIGHT, padx=5)
        
        # Stats frame
        stats_frame = ttk.Frame(details_frame)
        stats_frame.pack(fill=tk.X, pady=2)
        
        self.speed_var = tk.StringVar(value="Speed: --")
        self.size_var = tk.StringVar(value="Size: --")
        
        ttk.Label(stats_frame, textvariable=self.speed_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.size_var).pack(side=tk.RIGHT, padx=5)
        
        self.start_time = time.time()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.allow_close = False

    def on_closing(self):
        if self.allow_close:
            self.root.destroy()

    def update_progress(self, current_duration: float, total_duration: float, 
                       speed: str = "", size: str = "", status: str = None):
        """Update progress bar and stats."""
        if status:
            self.status_var.set(status)
        
        # Update progress bar
        progress = (current_duration / total_duration * 100) if total_duration > 0 else 0
        self.progress_var.set(progress)
        
        # Update elapsed time
        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        self.elapsed_var.set(f"Elapsed: {elapsed_str}")
        
        # Update estimated remaining time
        if progress > 0:
            total_estimated = elapsed / (progress / 100)
            remaining = total_estimated - elapsed
            remaining_str = time.strftime('%H:%M:%S', time.gmtime(remaining))
            self.remaining_var.set(f"Remaining: {remaining_str}")
        
        # Update speed and size
        if speed:
            self.speed_var.set(f"Speed: {speed}")
        if size:
            self.size_var.set(f"Size: {size}")
        
        # Update the window
        self.root.update()

    def close(self):
        """Allow and perform window close."""
        self.allow_close = True
        self.root.destroy()

class VideoMergerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Merger & YouTube Uploader")
        self.root.geometry("800x600")
        
        # Center the window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 800) // 2
        y = (screen_height - 600) // 2
        self.root.geometry(f"800x600+{x}+{y}")
        
        self.create_widgets()
        self.selected_files = []
        self.output_path = None
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Source folder selection
        ttk.Label(main_frame, text="Source Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.source_path = tk.StringVar()
        source_entry = ttk.Entry(main_frame, textvariable=self.source_path, width=60)
        source_entry.grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_source).grid(row=0, column=2)
        
        # Files listbox
        ttk.Label(main_frame, text="Available Videos:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.files_listbox = tk.Listbox(main_frame, width=70, height=15, selectmode=tk.MULTIPLE)
        self.files_listbox.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Scrollbar for listbox
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        scrollbar.grid(row=2, column=3, sticky=(tk.N, tk.S))
        self.files_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Output path selection
        ttk.Label(main_frame, text="Output Folder:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_folder = tk.StringVar()
        output_entry = ttk.Entry(main_frame, textvariable=self.output_folder, width=60)
        output_entry.grid(row=3, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=3, column=2)
        
        # YouTube upload options
        youtube_frame = ttk.LabelFrame(main_frame, text="YouTube Upload Options", padding="5")
        youtube_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(youtube_frame, text="Title:").grid(row=0, column=0, sticky=tk.W)
        self.video_title = tk.StringVar()
        ttk.Entry(youtube_frame, textvariable=self.video_title, width=60).grid(row=0, column=1, padx=5)
        
        ttk.Label(youtube_frame, text="Description:").grid(row=1, column=0, sticky=tk.W)
        self.video_desc = tk.Text(youtube_frame, width=45, height=3)
        self.video_desc.grid(row=1, column=1, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Merge Videos", command=self.merge_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Upload to YouTube", command=self.upload_to_youtube).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
    def browse_source(self):
        folder = filedialog.askdirectory(
            initialdir="\\\\fileserver2\\Head Office Server\\Projects Control (PC)\\10 Backup\\05 Tutorials\\Google\\Maps-Test",
            title="Select Source Folder"
        )
        if folder:
            self.source_path.set(folder)
            self.update_file_list()
            
    def browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder.set(folder)
            
    def update_file_list(self):
        self.files_listbox.delete(0, tk.END)
        folder = self.source_path.get()
        if folder:
            for file in sorted(glob.glob(os.path.join(folder, "**/*.mp4"), recursive=True)):
                self.files_listbox.insert(tk.END, os.path.relpath(file, folder))
                
    def merge_selected(self):
        selected_indices = self.files_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select videos to merge")
            return
            
        if not self.output_folder.get():
            messagebox.showwarning("Warning", "Please select output folder")
            return
            
        # Get selected files
        source_folder = self.source_path.get()
        video_files = []
        for idx in selected_indices:
            filename = self.files_listbox.get(idx)
            full_path = os.path.join(source_folder, filename)
            video_files.append((full_path, filename))
            
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_folder.get(), f"merged_{timestamp}.mp4")
        self.output_path = output_path
        
        # Start merge in a separate thread
        thread = threading.Thread(target=self._merge_thread, args=(video_files, output_path))
        thread.start()
        
    def _merge_thread(self, video_files: List[Tuple[str, str]], output_path: str):
        if merge_videos(video_files, output_path):
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                f"Videos merged successfully!\nSaved to: {output_path}"))
        else:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                "Failed to merge videos. Check the log file for details."))
                
    def get_credentials(self):
        creds = None
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            except Exception:
                pass
                
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(TOKEN_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
                
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
                
        return creds
        
    def upload_to_youtube(self):
        if not self.output_path or not os.path.exists(self.output_path):
            messagebox.showwarning("Warning", "Please merge videos first")
            return
            
        if not os.path.exists(TOKEN_FILE):
            messagebox.showerror("Error", 
                f"Missing {TOKEN_FILE}. Please obtain it from Google Cloud Console.")
            return
            
        title = self.video_title.get()
        description = self.video_desc.get("1.0", tk.END).strip()
        
        if not title:
            messagebox.showwarning("Warning", "Please enter a video title")
            return
            
        # Start upload in a separate thread
        thread = threading.Thread(target=self._upload_thread, args=(title, description))
        thread.start()
        
    def _upload_thread(self, title: str, description: str):
        try:
            creds = self.get_credentials()
            youtube = build('youtube', 'v3', credentials=creds)
            
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'categoryId': '22'  # People & Blogs
                },
                'status': {
                    'privacyStatus': 'private'
                }
            }
            
            insert_request = youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=MediaFileUpload(
                    self.output_path,
                    chunksize=-1,
                    resumable=True
                )
            )
            
            response = None
            while response is None:
                status, response = insert_request.next_chunk()
                if status:
                    self.root.after(0, lambda: messagebox.showinfo("Upload Status", 
                        f"Upload is {int(status.progress() * 100)}% complete..."))
                    
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                f"Upload Complete!\nVideo ID: {response['id']}"))
                
        except HttpError as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"An HTTP error occurred: {e.resp.status} {e.content}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                f"An error occurred: {str(e)}"))

def videos_are_compatible(video_paths: List[str]) -> bool:
    """Check if videos can be directly concatenated."""
    if not video_paths:
        return False
        
    try:
        # Get video info using ffprobe
        def get_video_info(path):
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,width,height,r_frame_rate',
                '-of', 'json',
                path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return json.loads(result.stdout)
            
        # Get first video's properties
        base_info = get_video_info(video_paths[0])
        base_stream = base_info['streams'][0]
        
        # Check all other videos
        for path in video_paths[1:]:
            info = get_video_info(path)
            stream = info['streams'][0]
            
            # Compare properties
            if (stream['codec_name'] != base_stream['codec_name'] or
                stream['width'] != base_stream['width'] or
                stream['height'] != base_stream['height'] or
                stream['r_frame_rate'] != base_stream['r_frame_rate']):
                return False
                
        return True
        
    except Exception as e:
        logging.error(f"Error checking video compatibility: {str(e)}")
        return False

def direct_concatenate(video_paths: List[str], output_path: str, progress_window) -> bool:
    """Merge videos using direct binary concatenation."""
    try:
        # Calculate total size for progress tracking
        total_size = sum(os.path.getsize(path) for path in video_paths)
        current_size = 0
        
        with open(output_path, 'wb') as outfile:
            for path in video_paths:
                # Update status
                filename = os.path.basename(path)
                progress_window.update_progress(
                    current_size,
                    total_size,
                    f"Processing {filename}..."
                )
                
                # Copy file in chunks
                with open(path, 'rb') as infile:
                    while True:
                        chunk = infile.read(1024*1024)  # 1MB chunks
                        if not chunk:
                            break
                        outfile.write(chunk)
                        current_size += len(chunk)
                        progress_window.update_progress(current_size, total_size)
                        
        return True
        
    except Exception as e:
        logging.error(f"Error in direct concatenation: {str(e)}")
        return False

def merge_with_moviepy(video_paths: List[str], output_path: str, progress_window) -> bool:
    """Merge videos using MoviePy."""
    clips = []
    total_duration = 0
    
    try:
        # First pass: load clips and calculate total duration
        for path in video_paths:
            filename = os.path.basename(path)
            progress_window.update_progress(
                0, 1,
                f"Loading {filename}..."
            )
            clip = VideoFileClip(path)
            clips.append(clip)
            total_duration += clip.duration
            
        # Second pass: concatenate and write
        final_clip = concatenate_videoclips(clips)
        
        def progress_callback(t):
            progress_window.update_progress(
                t,
                total_duration,
                "Writing merged video..."
            )
            
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            progress_bar=False,
            callback=progress_callback
        )
        
        return True
        
    except Exception as e:
        logging.error(f"Error in MoviePy merge: {str(e)}")
        return False
        
    finally:
        # Clean up clips
        for clip in clips:
            try:
                clip.close()
            except:
                pass

def merge_videos(video_files: List[Tuple[str, str]], output_path: str) -> bool:
    """Merge videos using MoviePy with terminal progress output."""
    try:
        if not video_files:
            return False
            
        print("\nStarting video merge process...")
        clips = []
        
        # Load all clips
        print("\nLoading video files:")
        for i, (video_path, filename) in enumerate(video_files, 1):
            print(f"  [{i}/{len(video_files)}] Loading: {filename}")
            clip = VideoFileClip(video_path)
            clips.append(clip)
            
        try:
            # Concatenate all clips
            print("\nConcatenating videos...")
            final_clip = concatenate_videoclips(clips)
            
            # Write final video
            print(f"\nWriting merged video to: {output_path}")
            print("This may take a while... Please wait...")
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                logger=None  # Disable progress bar
            )
            print("\nMerge completed successfully!")
            return True
            
        finally:
            # Clean up clips
            print("\nCleaning up...")
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
                    
    except Exception as e:
        print(f"\nError during merge: {str(e)}")
        logging.error(f"Error in merge_videos: {str(e)}", exc_info=True)
        return False

def test_merge():
    """Test function to try out video merging."""
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for MP4 files in the current directory
    video_files = []
    for file in sorted(glob.glob(os.path.join(current_dir, "*.mp4"))):
        # Create tuple of (full_path, filename)
        video_files.append((file, os.path.basename(file)))
    
    if not video_files:
        print("No MP4 files found in the current directory")
        return
    
    print(f"Found {len(video_files)} MP4 files:")
    for _, filename in video_files:
        print(f"- {filename}")
    
    # Create output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(current_dir, f"merged_{timestamp}.mp4")
    
    print(f"\nAttempting to merge videos into: {output_path}")
    if merge_videos(video_files, output_path):
        print(f"\nSuccess! Merged video saved to: {output_path}")
    else:
        print("\nError: Video merge failed. Check the log file for details.")

def get_folder_path_gui() -> str:
    """Create a Tkinter window to get the folder path."""
    try:
        print("\nPlease select the tutorials folder...")
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        folder_path = filedialog.askdirectory(
            title="Select Tutorials Folder",
            initialdir="\\\\fileserver2\\Head Office Server\\Projects Control (PC)\\10 Backup\\05 Tutorials\\Google\\Maps-Test"
        )
        
        return folder_path.strip() if folder_path else ""
        
    except Exception as e:
        logging.error(f"Error in folder selection: {str(e)}")
        return ""

def upload_to_youtube(youtube, video_path: str, title: str, description: str) -> Optional[Dict]:
    """Upload video to YouTube with terminal progress output."""
    try:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return None
            
        print(f"\nPreparing to upload: {title}")
        
        # Disable SSL verification
        ssl._create_default_https_context = ssl._create_unverified_context
        
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': ['tutorial'],
                'categoryId': '27'  # Education category
            },
            'status': {
                'privacyStatus': 'private',
                'selfDeclaredMadeForKids': False
            }
        }
        
        print("\nStarting upload...")
        file_size = os.path.getsize(video_path)
        
        insert_request = youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=MediaFileUpload(
                video_path, 
                chunksize=1024*1024,  # 1MB chunks
                resumable=True
            )
        )
        
        start_time = time.time()
        response = None
        
        while response is None:
            try:
                status, response = insert_request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    current_size = int(file_size * status.progress())
                    elapsed = time.time() - start_time
                    speed = current_size / (1024 * 1024 * elapsed)  # MB/s
                    
                    # Print progress to terminal
                    print(f"\rProgress: {progress}% ({speed:.1f} MB/s)", end="")
                    
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    # Retry on server errors
                    time.sleep(5)
                else:
                    raise
                    
        print(f"\nUpload successful! Video ID: {response['id']}")
        return response
        
    except HttpError as e:
        print(f"An HTTP error occurred: {e.resp.status} {e.content}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def authenticate_youtube(client_secrets_file: str) -> Optional[object]:
    """Authenticate with YouTube as the very first step."""
    try:
        print("\nChecking YouTube authentication...")
        if not os.path.exists(client_secrets_file):
            print("Error: Client secrets file not found")
            return None
            
        # Disable SSL verification
        ssl._create_default_https_context = ssl._create_unverified_context
        
        creds = None
        # Check if token file exists
        if os.path.exists(TOKEN_FILE):
            try:
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
                print("Found existing token")
            except Exception as e:
                print(f"Error loading token: {e}")
                creds = None

        # If token is missing or invalid, refresh or re-authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    print("Token expired, refreshing...")
                    creds.refresh(Request())
                    print("Token refreshed successfully")
                except Exception as e:
                    print(f"Token refresh failed: {e}")
                    creds = None

            if not creds:
                print("\nOpening browser for YouTube authentication...")
                print("Please select your Google account and grant access...")
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
                creds = flow.run_local_server(port=0)

                # Save the new token
                with open(TOKEN_FILE, "w") as token:
                    token.write(creds.to_json())
                print("New token saved")

        # Build and return the YouTube service
        youtube = build("youtube", "v3", credentials=creds)
        print("YouTube authentication successful!")
        return youtube
        
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def get_structured_title(path: str) -> str:
    """Generate a structured title from the path levels after '05 Tutorials'."""
    try:
        parts = Path(path).parts
        # Find the index of "05 Tutorials"
        for i, part in enumerate(parts):
            if part == "05 Tutorials":
                # Take exactly 4 parts after "05 Tutorials"
                relevant_parts = parts[i+1:i+5]  # This will get: Google, Slides, Lynda, Google Slides Essential Training
                # Join parts and ensure total length is under 100 characters
                title = " - ".join(relevant_parts)
                return title[:100]  # YouTube title limit is 100 characters
    except Exception as e:
        logging.error(f"Error generating title: {str(e)}")
        return "Untitled Video"
    return "Untitled Video"

def is_tutorial_folder(path: str) -> bool:
    """Check if the given path is a tutorial folder."""
    try:
        # Check if the path exists and is a directory
        if not os.path.isdir(path):
            return False
            
        # Check if there are any MP4 files in the directory
        mp4_files = glob.glob(os.path.join(path, "*.mp4"))
        return len(mp4_files) > 0
        
    except Exception as e:
        logging.error(f"Error checking tutorial folder: {str(e)}")
        return False

def find_tutorial_folders(parent_path: str) -> List[str]:
    """Find all tutorial folders under the parent path."""
    tutorial_folders = []
    
    try:
        # First level: Company folders (e.g., Udemy, LinkedIn)
        for company_dir in os.listdir(parent_path):
            company_path = os.path.join(parent_path, company_dir)
            if not os.path.isdir(company_path):
                continue
                
            print(f"\nChecking company folder: {company_dir}")
            
            # Second level: Tutorial folders
            for tutorial_dir in os.listdir(company_path):
                tutorial_path = os.path.join(company_path, tutorial_dir)
                if not os.path.isdir(tutorial_path):
                    continue
                    
                # Skip if already processed
                if os.path.exists(os.path.join(tutorial_path, "youtube_link.txt")):
                    print(f"  Skipping {tutorial_dir} (already processed)")
                    continue
                    
                # Check if this folder or its subfolders contain MP4 files
                has_videos = False
                for root, _, files in os.walk(tutorial_path):
                    if any(f.endswith('.mp4') for f in files):
                        has_videos = True
                        break
                        
                if has_videos:
                    print(f"  Found tutorial: {tutorial_dir}")
                    tutorial_folders.append(tutorial_path)
                    
    except Exception as e:
        print(f"Error finding tutorial folders: {e}")
        
    return tutorial_folders

def collect_videos(folder_path: str) -> List[Tuple[str, str]]:
    """Collect all MP4 files in the folder and its subfolders."""
    try:
        video_files = []
        print(f"\nSearching for videos in: {folder_path}")
        
        # First check for videos directly in the tutorial folder
        direct_mp4s = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]
        if direct_mp4s:
            print(f"Found {len(direct_mp4s)} videos in tutorial folder:")
            for file in sorted(direct_mp4s):
                full_path = os.path.join(folder_path, file)
                video_files.append((full_path, file))
                print(f"  - {file}")
        
        # Then check all subfolders
        for root, dirs, files in os.walk(folder_path):
            # Skip the tutorial folder itself as we already checked it
            if root == folder_path:
                continue
                
            # Sort files to maintain order
            mp4_files = [f for f in files if f.lower().endswith('.mp4')]
            if mp4_files:
                rel_path = os.path.relpath(root, folder_path)
                print(f"\nFound {len(mp4_files)} videos in subfolder '{rel_path}':")
                for file in sorted(mp4_files):
                    full_path = os.path.join(root, file)
                    # Store the relative path in the filename for better organization
                    display_name = f"{os.path.basename(rel_path)} - {file}"
                    video_files.append((full_path, display_name))
                    print(f"  - {file}")
                    
        if not video_files:
            print("No MP4 files found in tutorial folder or subfolders")
        else:
            print(f"\nTotal videos found: {len(video_files)}")
            
        return video_files
    except Exception as e:
        logging.error(f"Error collecting videos: {str(e)}")
        return []

def get_video_duration(video_path: str) -> Optional[float]:
    """Get the duration of a video file using MoviePy."""
    try:
        with VideoFileClip(video_path) as clip:
            return clip.duration
    except Exception as e:
        logging.error(f"Error getting video duration: {str(e)}")
        return None

def generate_timestamps(video_files: List[Tuple[str, str]]) -> str:
    """Generate timestamps for video description."""
    try:
        timestamps = []
        current_time = 0
        
        for video_path, filename in video_files:
            duration = get_video_duration(video_path)
            if duration is None:
                continue
                
            # Format current time as HH:MM:SS
            time_str = time.strftime('%H:%M:%S', time.gmtime(current_time))
            
            # Extract video number and name from filename
            match = re.match(r'(\d+).*?([^.]+)\.mp4$', filename)
            if match:
                video_num, video_name = match.groups()
                timestamps.append(f"{time_str} - {video_num}. {video_name}")
            else:
                timestamps.append(f"{time_str} - {filename}")
                
            current_time += duration
            
        return "\n".join(timestamps)
        
    except Exception as e:
        logging.error(f"Error generating timestamps: {str(e)}")
        return ""

def merge_videos_in_batches(video_files: List[Tuple[str, str]], output_path: str, 
                          batch_size: int = 5, progress_window: Optional[ProgressWindow] = None) -> bool:
    """Merge videos in batches using MoviePy."""
    try:
        if not video_files:
            return False
            
        # Calculate total duration for progress tracking
        total_duration = 0
        for video_path, _ in video_files:
            duration = get_video_duration(video_path)
            if duration:
                total_duration += duration
                
        if total_duration == 0:
            return False
            
        # Create batches
        batches = [video_files[i:i + batch_size] for i in range(0, len(video_files), batch_size)]
        temp_files = []
        current_duration = 0
        
        # Process each batch
        for batch_num, batch in enumerate(batches):
            batch_output = f"temp_batch_{batch_num}.mp4"
            temp_files.append(batch_output)
            
            # Load clips
            clips = []
            try:
                for video_path, filename in batch:
                    if progress_window:
                        progress_window.update_progress(
                            current_duration,
                            total_duration,
                            status=f"Loading {filename}..."
                        )
                    clip = VideoFileClip(video_path)
                    clips.append(clip)
                    
                # Concatenate batch
                if progress_window:
                    progress_window.update_progress(
                        current_duration,
                        total_duration,
                        status=f"Merging batch {batch_num + 1}/{len(batches)}..."
                    )
                    
                final_clip = concatenate_videoclips(clips)
                
                # Write the merged clip
                if progress_window:
                    progress_window.update_progress(
                        current_duration,
                        total_duration,
                        status=f"Writing batch {batch_num + 1}/{len(batches)}..."
                    )
                
                final_clip.write_videofile(
                    batch_output,
                    codec='libx264',
                    audio_codec='aac',
                    logger=None  # Disable progress bar
                )
                
                # Update progress based on batch duration
                current_duration += sum(clip.duration for clip in clips)
                if progress_window:
                    progress_window.update_progress(
                        current_duration,
                        total_duration
                    )
                
            finally:
                # Clean up clips
                for clip in clips:
                    try:
                        clip.close()
                    except:
                        pass
                        
        # If there's only one batch, rename it to the final output
        if len(temp_files) == 1:
            shutil.move(temp_files[0], output_path)
        else:
            # Merge all batches
            if progress_window:
                progress_window.update_progress(
                    current_duration,
                    total_duration,
                    status="Merging final video..."
                )
                
            clips = []
            try:
                for temp_file in temp_files:
                    clip = VideoFileClip(temp_file)
                    clips.append(clip)
                    
                final_clip = concatenate_videoclips(clips)
                
                # Write the final merged clip
                if progress_window:
                    progress_window.update_progress(
                        current_duration,
                        total_duration,
                        status="Writing final video..."
                    )
                
                final_clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    logger=None  # Disable progress bar
                )
                
            finally:
                # Clean up clips and temp files
                for clip in clips:
                    try:
                        clip.close()
                    except:
                        pass
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
        return True
        
    except Exception as e:
        logging.error(f"Error in merge_videos_in_batches: {str(e)}")
        return False

def cleanup_and_save_link(folder_path: str, video_id: str, title: str):
    """Save video link to a file and clean up."""
    link_file = os.path.join(folder_path, "youtube_link.txt")
    with open(link_file, "w") as f:
        f.write(f"Title: {title}\n")
        f.write(f"Video ID: {video_id}\n")
        f.write(f"URL: https://youtu.be/{video_id}\n")

def delete_video_folders(tutorial_path: str) -> bool:
    """Delete video folders after successful upload."""
    try:
        # Keep youtube_link.txt
        link_file = os.path.join(tutorial_path, "youtube_link.txt")
        if os.path.exists(link_file):
            temp_link = link_file + ".tmp"
            shutil.copy2(link_file, temp_link)
            
        # Delete all files and folders
        for item in os.listdir(tutorial_path):
            item_path = os.path.join(tutorial_path, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")
                
        # Restore youtube_link.txt
        if os.path.exists(temp_link):
            shutil.move(temp_link, link_file)
            
        return True
        
    except Exception as e:
        print(f"Error in delete_video_folders: {e}")
        return False

def shorten_folder_name(folder_name: str) -> str:
    """Shorten folder name to avoid path length issues."""
    if len(folder_name) <= 50:
        return folder_name
        
    # Keep first 25 and last 24 characters
    return folder_name[:25] + "..." + folder_name[-24:]

def create_temp_folder_with_videos(original_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Create a temporary folder with shortened paths for video processing."""
    try:
        # Create temp directory inside the tutorial folder
        temp_dir = os.path.join(original_path, "temp_processing")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create unique subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_folder = os.path.join(temp_dir, f"merge_{timestamp}")
        os.makedirs(temp_folder)
        
        # Collect all videos first
        video_files = collect_videos(original_path)
        if not video_files:
            return None, None
            
        print("\nCopying videos to temporary folder...")
        # Copy videos with shortened names
        for i, (video_path, original_name) in enumerate(video_files, 1):
            # Create shortened name with index to maintain order
            new_name = f"{i:02d}_{shorten_folder_name(original_name)}"
            
            # Copy file
            new_path = os.path.join(temp_folder, new_name)
            print(f"  {i:02d}. {original_name} -> {new_name}")
            shutil.copy2(video_path, new_path)
            
        return temp_folder, original_path
        
    except Exception as e:
        print(f"Error creating temp folder: {e}")
        return None, None

def cleanup_temp_folder(temp_folder: str):
    """Clean up temporary processing folder."""
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
    except Exception as e:
        print(f"Error cleaning up temp folder: {e}")

def process_tutorial_folder(tutorial_path: str, youtube: object):
    """Process a single tutorial folder."""
    try:
        print(f"\nProcessing folder: {tutorial_path}")
        
        # Check if already processed
        link_file = os.path.join(tutorial_path, "youtube_link.txt")
        if os.path.exists(link_file):
            print("Folder already processed (youtube_link.txt exists)")
            return True
            
        # Create temporary folder with shortened paths
        print("\nCreating temporary folder for processing...")
        temp_folder, original_path = create_temp_folder_with_videos(tutorial_path)
        if not temp_folder:
            print("Failed to create temporary folder")
            return False
            
        try:
            # Collect videos from temp folder
            print("\nCollecting videos...")
            video_files = collect_videos(temp_folder)
            if not video_files:
                print("No video files found")
                return False
                
            print(f"\nFound {len(video_files)} videos to merge")
            
            # Create output filename in the tutorial folder
            output_filename = "merged.mp4"
            output_path = os.path.join(tutorial_path, output_filename)
            
            # Merge videos
            print("\nMerging videos...")
            if not merge_videos(video_files, output_path):
                print("Failed to merge videos")
                return False
                
            print("Videos merged successfully")
            
            # Generate video title
            title = get_structured_title(original_path)
            print(f"\nVideo title: {title}")
            
            # Generate description with timestamps
            description = generate_timestamps(video_files)
            if description:
                print("\nGenerated timestamps for description")
                print("Description preview:")
                print("-------------------")
                print(description[:500] + "..." if len(description) > 500 else description)
                print("-------------------")
            
            # Upload to YouTube
            print("\nUploading to YouTube...")
            response = upload_to_youtube(youtube, output_path, title, description)
            
            if response and 'id' in response:
                video_id = response['id']
                print(f"Upload successful! Video ID: {video_id}")
                
                # Save link and clean up
                cleanup_and_save_link(original_path, video_id, title)
                delete_video_folders(original_path)
                print("Folder cleaned up")
                return True
            else:
                print("Upload failed")
                return False
                
        finally:
            # Clean up temp folder
            cleanup_temp_folder(temp_folder)
            
    except Exception as e:
        print(f"Error processing tutorial folder: {e}")
        logging.error(f"Error in process_tutorial_folder: {str(e)}", exc_info=True)
        return False

def process_folder(parent_path: str, youtube: object):
    """Process all tutorial folders under the parent path."""
    # Find all tutorial folders
    print(f"\nScanning software folder: {parent_path}")
    tutorial_folders = find_tutorial_folders(parent_path)
    
    if not tutorial_folders:
        print("\nNo tutorial folders found")
        return
        
    print(f"\nFound {len(tutorial_folders)} tutorial folders to process")
    
    # Process each folder
    for i, folder in enumerate(tutorial_folders, 1):
        print(f"\nProcessing tutorial {i}/{len(tutorial_folders)}")
        print(f"Folder: {os.path.basename(folder)}")
        process_tutorial_folder(folder, youtube)

def get_script_directory():
    """Get the directory containing the script."""
    try:
        # Get script path
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            script_path = sys.executable
        else:
            # Running as script
            script_path = os.path.abspath(__file__)
            
        # Get directory
        script_dir = os.path.dirname(script_path)
        
        # Create if doesn't exist
        os.makedirs(script_dir, exist_ok=True)
        
        return script_dir
        
    except Exception as e:
        print(f"Error getting script directory: {e}")
        return os.getcwd()

def find_client_secrets():
    """Find the client_secrets.json file using GUI if not found in default locations."""
    try:
        # Check script directory first (04.merge.and.upload)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        secrets_path = os.path.join(script_dir, "client_secrets.json")
        if os.path.exists(secrets_path):
            return secrets_path
            
        # Check current directory
        if os.path.exists("client_secrets.json"):
            return os.path.abspath("client_secrets.json")
            
        # If not found, show file picker dialog
        print("\nClient secrets file not found in default locations.")
        print("Please select your client_secrets.json file...")
        
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select client_secrets.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=script_dir
        )
        
        if file_path and os.path.exists(file_path):
            # Copy the file to the script directory
            try:
                shutil.copy2(file_path, secrets_path)
                print(f"Copied client_secrets.json to: {script_dir}")
                return secrets_path
            except Exception as e:
                print(f"Error copying file: {e}")
                return file_path
        
        return None
        
    except Exception as e:
        print(f"Error finding client secrets: {e}")
        return None

def main():
    try:
        print("YouTube Video Merger and Uploader")
        print("=================================")
        
        # Find client secrets file
        print("\nLooking for client_secrets.json...")
        client_secrets_file = find_client_secrets()
        if not client_secrets_file:
            print("Error: client_secrets.json not found")
            return
        print(f"Using client secrets from: {client_secrets_file}")
            
        # Authenticate with YouTube
        print("\nAuthenticating with YouTube...")
        youtube = authenticate_youtube(client_secrets_file)
        if not youtube:
            print("YouTube authentication failed")
            return
            
        # Get folder path from user
        print("\nWaiting for folder selection...")
        folder_path = get_folder_path_gui()
        if not folder_path:
            print("No folder path provided")
            return
        print(f"Selected folder: {folder_path}")
            
        # Process the folder
        print("\nScanning for tutorial folders...")
        process_folder(folder_path, youtube)
        
        print("\nProcessing complete!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        
    finally:
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()