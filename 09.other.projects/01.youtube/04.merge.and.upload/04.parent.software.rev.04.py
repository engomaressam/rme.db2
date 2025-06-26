#!/usr/bin/env python3
import os
import glob
import json
import time
import shutil
import logging
import re
from datetime import datetime
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import ffmpeg
import tkinter as tk
from tkinter import ttk, messagebox
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

def get_folder_path_gui() -> str:
    """Create a Tkinter window to get the folder path."""
    try:
        def on_submit():
            window.quit()

        # Create the main window
        window = tk.Tk()
        window.title("Enter Folder Path")
        window.geometry("800x150")  # Make window wider for long paths
        
        # Create and pack a frame
        frame = ttk.Frame(window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create label
        ttk.Label(frame, text="Enter the full path to the tutorials folder:").pack(pady=5)
        
        # Create entry field
        path_var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=path_var, width=100)  # Make entry field wider
        entry.pack(pady=5, fill=tk.X)
        
        # Create submit button
        ttk.Button(frame, text="Submit", command=on_submit).pack(pady=10)
        
        # Focus on entry field and bind Enter key
        entry.focus()
        window.bind('<Return>', lambda e: on_submit())
        
        # Bring window to front
        window.lift()
        window.attributes('-topmost', True)
        window.attributes('-topmost', False)
        
        # Start the window
        window.mainloop()
        
        # Get the path and destroy the window
        path = path_var.get().strip()
        window.destroy()
        return path
        
    except Exception as e:
        logging.error(f"Error creating GUI window: {str(e)}")
        # Fallback to console input if GUI fails
        return input("Enter folder path (GUI failed): ")

def authenticate_youtube(client_secrets_file: str) -> Optional[object]:
    """Authenticate with YouTube as the very first step."""
    try:
        print("\nChecking YouTube authentication...")
        if not os.path.exists(client_secrets_file):
            print("Error: Client secrets file not found")
            return None
            
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
                # Take all parts after "05 Tutorials"
                relevant_parts = parts[i+1:]
                # Filter out empty parts but keep "Lynda"
                title_parts = []
                for part in relevant_parts:
                    # Skip empty parts
                    if not part:
                        continue
                    # Include the part if it's "Lynda" or not a system folder
                    if part == "Lynda" or not part.startswith('.'):
                        title_parts.append(part)
                
                # Join parts and ensure total length is under 100 characters
                title = " - ".join(title_parts)
                # Truncate if necessary while keeping the structure
                if len(title) > 95:
                    title = title[:95] + "..."
                return title
                
        # Fallback to just the folder name
        return os.path.basename(path)
    except Exception as e:
        logging.error(f"Error generating title: {str(e)}")
        return os.path.basename(path)  # Fallback to just the folder name

def is_tutorial_folder(path: str) -> bool:
    """
    Check if this is a tutorial folder.
    A tutorial folder should be the main course folder containing all videos,
    not its parent folders or subfolders.
    """
    try:
        parts = Path(path).parts
        
        # Find "05 Tutorials" in the path
        tutorial_index = -1
        for i, part in enumerate(parts):
            if part == "05 Tutorials":
                tutorial_index = i
                break
                
        if tutorial_index == -1:
            return False
            
        # Count levels after "05 Tutorials"
        levels_after = len(parts) - (tutorial_index + 1)
        
        # We expect: Google/Slides/Lynda/Course Name
        # So 4 levels after "05 Tutorials"
        if levels_after != 4:
            return False
            
        # Check if this folder has video files in its structure
        has_videos = False
        has_subfolders = False
        
        for root, dirs, files in os.walk(path):
            # If we find numbered subfolders, this is likely the main course folder
            if any(re.match(r'^\d+\.', d) for d in dirs):
                has_subfolders = True
                
            # Count video files
            if any(f.lower().endswith(('.mp4', '.avi', '.mkv')) for f in files):
                has_videos = True
                
            if has_videos and has_subfolders:
                return True
                
        return False
        
    except Exception as e:
        logging.error(f"Error in is_tutorial_folder: {str(e)}")
        return False

def collect_videos(folder_path: str) -> List[Tuple[str, str]]:
    """Collect all videos from folder and subfolders, sorted by folder structure and filename."""
    video_files = []
    try:
        # Walk through all subdirectories
        for root, _, files in os.walk(folder_path):
            # Filter and sort video files
            video_files.extend(
                (os.path.join(root, f), os.path.relpath(os.path.join(root, f), folder_path))
                for f in sorted(files)
                if f.lower().endswith(('.mp4', '.avi', '.mkv'))
            )
        
        # Sort by full path to maintain folder structure order
        return sorted(video_files, key=lambda x: x[0].lower())
    except Exception as e:
        logging.error(f"Error collecting videos: {str(e)}")
        return []

def get_video_duration(video_path: str) -> Optional[float]:
    """Get video duration using ffprobe with better error handling and retries."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", 
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", 
                 video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=30  # Add timeout
            )
            duration = float(result.stdout.strip())
            return duration
        except subprocess.TimeoutExpired:
            logging.warning(f"Timeout getting duration for {os.path.basename(video_path)}, attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)  # Wait before retry
        except Exception as e:
            logging.warning(f"Could not get duration for {os.path.basename(video_path)}: {str(e)}")
            return None

def generate_timestamps(video_files: List[Tuple[str, str]]) -> str:
    """Generate timestamps for video description with proper duration handling."""
    timestamps = []
    current_time = 0
    
    for full_path, rel_path in video_files:
        duration = get_video_duration(full_path)
        if duration is None:
            logging.warning(f"Skipping timestamp for {rel_path} - could not get duration")
            continue
            
        hours = int(current_time // 3600)
        minutes = int((current_time % 3600) // 60)
        seconds = int(current_time % 60)
        
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        video_name = os.path.splitext(os.path.basename(rel_path))[0]
        timestamps.append(f"{timestamp} - {video_name}")
        
        current_time += duration
    
    return '\n'.join(timestamps)

def parse_ffmpeg_progress(line: str) -> Dict[str, str]:
    """Parse ffmpeg progress output line."""
    result = {}
    
    # Extract time
    time_match = re.search(r'time=(\d+:\d+:\d+\.\d+)', line)
    if time_match:
        time_str = time_match.group(1)
        h, m, s = time_str.split(':')
        s = float(s)
        result['duration'] = float(h) * 3600 + float(m) * 60 + s
    
    # Extract speed
    speed_match = re.search(r'speed=(\d+\.?\d*x)', line)
    if speed_match:
        result['speed'] = speed_match.group(1)
    
    # Extract size
    size_match = re.search(r'size=\s*(\d+.*?B)', line)
    if size_match:
        result['size'] = size_match.group(1)
    
    return result

def merge_videos(video_files: List[Tuple[str, str]], output_path: str) -> bool:
    """Merge videos using ffmpeg with proper format settings and UI progress bar."""
    progress_window = None
    try:
        # Create temporary directory for intermediate files
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_merge")
        os.makedirs(temp_dir, exist_ok=True)
        
        # First pass: Convert all videos to same format
        logging.info(f"\nAnalyzing {len(video_files)} videos for merge:")
        total_duration = 0
        standardized_files = []
        
        for i, (full_path, rel_path) in enumerate(video_files, 1):
            logging.info(f"Video {i}/{len(video_files)}: {rel_path}")
            
            # Get duration
            duration = get_video_duration(full_path) or 0
            total_duration += duration
            
            # Create standardized filename
            std_filename = f"part_{i:03d}.mp4"
            std_path = os.path.join(temp_dir, std_filename)
            standardized_files.append(std_path)
            
            # Convert to standard format
            cmd = [
                "ffmpeg",
                "-i", full_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-y",
                std_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Error standardizing video {rel_path}: {result.stderr}")
                return False
        
        if total_duration == 0:
            logging.error("Could not calculate total duration of videos")
            return False
            
        # Create concat file
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for std_file in standardized_files:
                f.write(f"file '{os.path.basename(std_file)}'\n")
        
        # Create progress window
        progress_window = ProgressWindow("Merging Videos")
        progress_window.update_progress(0, total_duration, status="Starting merge process...")
        
        # Second pass: Concatenate standardized files
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",  # Just copy streams, no re-encoding
            "-movflags", "+faststart",
            "-y",
            output_path
        ]
        
        # Run ffmpeg with progress monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            cwd=temp_dir  # Set working directory to temp_dir
        )
        
        # Monitor the encoding progress
        error_output = []
        while True:
            stderr_line = process.stderr.readline()
            if not stderr_line and process.poll() is not None:
                break
                
            error_output.append(stderr_line)
            
            if "time=" in stderr_line:
                progress_data = parse_ffmpeg_progress(stderr_line)
                if progress_data:
                    current_duration = progress_data.get('duration', 0)
                    speed = progress_data.get('speed', '')
                    size = progress_data.get('size', '')
                    
                    if progress_window:
                        status = f"Merging videos... ({current_duration:.1f}s / {total_duration:.1f}s)"
                        progress_window.update_progress(
                            current_duration,
                            total_duration,
                            speed=speed,
                            size=size,
                            status=status
                        )
        
        # Get the final status
        return_code = process.wait()
        
        if return_code != 0:
            error_msg = '\n'.join(error_output)
            logging.error(f"FFmpeg error: {error_msg}")
            if progress_window:
                progress_window.update_progress(0, total_duration, status="Error in merge process!")
                time.sleep(3)
            return False
            
        # Verify output duration
        output_duration = get_video_duration(output_path)
        if output_duration is None:
            logging.error("Could not verify output video duration")
            return False
            
        # Allow 5% tolerance for duration mismatch
        duration_diff = abs(output_duration - total_duration)
        duration_tolerance = total_duration * 0.05
        
        if duration_diff > duration_tolerance:
            logging.error(f"Output duration ({output_duration:.2f}s) doesn't match expected ({total_duration:.2f}s)")
            return False
            
        if progress_window:
            progress_window.update_progress(total_duration, total_duration, status="Merge completed successfully!")
            time.sleep(2)
            
        logging.info(f"Merge completed successfully! Final video duration: {output_duration:.2f} seconds")
        return True
        
    except Exception as e:
        logging.error(f"Error merging videos: {str(e)}")
        if progress_window:
            progress_window.update_progress(0, total_duration, status=f"Error: {str(e)}")
            time.sleep(3)
        return False
    finally:
        if progress_window:
            progress_window.close()
        # Clean up temp directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass

def upload_to_youtube(youtube, video_path: str, title: str, description: str) -> Optional[Dict]:
    """Upload video to YouTube with improved error handling and retry logic."""
    try:
        # Ensure title is not empty and within limits
        if not title or len(title) > 100:
            title = title[:95] if title else "Untitled Video"
        
        logging.info(f"Starting upload: {title}")
        
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": "27",  # Education category
                "tags": ["tutorial", "education"]
            },
            "status": {
                "privacyStatus": "unlisted",
                "selfDeclaredMadeForKids": False
            }
        }

        # Create insert request
        insert_request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=MediaFileUpload(
                video_path, 
                chunksize=1024*1024*8,  # 8MB chunks for better reliability
                resumable=True
            )
        )

        response = None
        retries = 3
        chunk_retries = 5  # Number of retries per chunk
        
        while response is None and retries > 0:
            try:
                chunk_retry_count = chunk_retries
                while chunk_retry_count > 0:
                    try:
                        status, response = insert_request.next_chunk()
                        if response:
                            logging.info("Upload completed successfully")
                            return response
                        elif status:
                            logging.info(f"Upload progress: {int(status.progress() * 100)}%")
                        break  # Successful chunk upload, continue to next chunk
                    except (ConnectionError, httplib2.HttpLib2Error) as chunk_error:
                        chunk_retry_count -= 1
                        if chunk_retry_count == 0:
                            raise  # Re-raise if we're out of chunk retries
                        logging.warning(f"Chunk upload failed, waiting 10 seconds before retry... ({str(chunk_error)})")
                        time.sleep(10)  # Longer wait between chunk retries
                
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise
                logging.warning(f"Upload attempt failed, waiting 30 seconds before retry... ({str(e)})")
                time.sleep(30)  # Much longer wait between full upload retries

    except Exception as e:
        logging.error(f"Error uploading to YouTube: {str(e)}")
        return None

def cleanup_and_save_link(folder_path: str, video_id: str) -> None:
    """Save the YouTube link to a file and clean up temporary files."""
    try:
        # Create the YouTube link
        youtube_link = f"https://youtu.be/{video_id}"
        
        # Save the link in the tutorial folder
        link_file_path = os.path.join(folder_path, "youtube link.txt")
        with open(link_file_path, "w") as f:
            f.write(youtube_link)
        logging.info(f"Saved YouTube link to: {link_file_path}")
        
        # Clean up any temporary files if they exist
        temp_files = ["temp_merged.mp4", "temp_output.mp4"]
        for temp_file in temp_files:
            temp_path = os.path.join(folder_path, temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logging.info(f"Cleaned up temporary file: {temp_path}")
    except Exception as e:
        logging.error(f"Error in cleanup_and_save_link: {str(e)}")
        raise

def delete_video_folders(tutorial_path: str) -> bool:
    """Delete all subfolders containing videos after successful upload."""
    try:
        deleted_folders = 0
        # Walk through all subfolders
        for root, dirs, files in os.walk(tutorial_path, topdown=False):
            # Skip the tutorial root folder itself
            if root == tutorial_path:
                continue
                
            # Check if this is a folder containing videos
            has_videos = any(f.lower().endswith(('.mp4', '.avi', '.mkv')) for f in files)
            if has_videos:
                try:
                    shutil.rmtree(root)
                    deleted_folders += 1
                    logging.info(f"Deleted folder: {root}")
                except Exception as e:
                    logging.error(f"Failed to delete folder {root}: {str(e)}")
                    return False
        
        logging.info(f"Deleted {deleted_folders} folders containing videos")
        return True
    except Exception as e:
        logging.error(f"Error deleting video folders: {str(e)}")
        return False

def shorten_folder_name(folder_name: str) -> str:
    """Create a shorter version of the folder name that's still meaningful."""
    # Remove special characters and spaces
    cleaned = re.sub(r'[^\w\s-]', '', folder_name)
    # Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')
    # Take first word of each segment
    words = [part.split('_')[0] for part in cleaned.split('-')]
    # Join with underscores and limit to 50 characters
    shortened = '_'.join(words)[:50]
    return shortened

def create_temp_folder_with_videos(original_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Creates a temporary folder with a shorter name and copies videos there.
    Returns tuple of (temp_folder_path, original_folder_name) or (None, None) on failure.
    """
    try:
        # Get original folder name and create shortened version
        original_folder_name = os.path.basename(original_path)
        short_name = shorten_folder_name(original_folder_name)
        
        # Create temp folder in parent directory
        parent_dir = os.path.dirname(original_path)
        temp_folder = os.path.join(parent_dir, f"temp_{short_name}")
        
        # If temp folder already exists, delete it
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        
        # Create new temp folder and copy structure
        os.makedirs(temp_folder)
        
        # Copy all video files maintaining relative paths
        for root, _, files in os.walk(original_path):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mkv')):
                    # Get relative path
                    rel_path = os.path.relpath(root, original_path)
                    # Create corresponding folder in temp directory
                    temp_subdir = os.path.join(temp_folder, rel_path)
                    os.makedirs(temp_subdir, exist_ok=True)
                    # Copy file
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(temp_subdir, file)
                    shutil.copy2(src_file, dst_file)
        
        return temp_folder, original_folder_name
    except Exception as e:
        logging.error(f"Error creating temporary folder: {str(e)}")
        return None, None

def cleanup_temp_folder(temp_folder: str):
    """Clean up the temporary folder."""
    try:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            logging.info(f"Cleaned up temporary folder: {temp_folder}")
    except Exception as e:
        logging.warning(f"Failed to clean up temporary folder: {str(e)}")

def process_tutorial_folder(tutorial_path: str, youtube: object):
    """Process a single tutorial folder containing videos."""
    merged_path = None
    temp_folder = None
    try:
        logging.info(f"\nProcessing tutorial folder: {tutorial_path}")
        
        # Check if this folder has already been processed
        link_file = os.path.join(tutorial_path, "youtube link.txt")
        if os.path.exists(link_file):
            logging.info(f"Folder already processed (found youtube link.txt). Skipping: {tutorial_path}")
            return
        
        # Get videos from working path
        video_files = collect_videos(tutorial_path)
        if not video_files:
            logging.error(f"No video files found in {tutorial_path}")
            return
            
        logging.info(f"Found {len(video_files)} videos:")
        for _, rel_path in video_files:
            logging.info(f"  - {rel_path}")
        
        # Get structured title
        title = get_structured_title(tutorial_path)
        logging.info(f"Generated title: {title}")
        
        # Create merged file in the tutorial folder
        folder_name = os.path.basename(tutorial_path)
        safe_name = re.sub(r'[^\w\s-]', '', folder_name).strip()[:50]
        merged_filename = f"{safe_name}_merged.mp4"
        merged_path = os.path.join(tutorial_path, merged_filename)
        
        logging.info("Starting video merge...")
        if not merge_videos(video_files, merged_path):
            logging.error("Failed to merge videos")
            return
            
        logging.info("Videos merged successfully")
        
        # Generate timestamps
        timestamps = generate_timestamps(video_files)
        description = "Tutorial Contents:\n\n" + timestamps
            
        # Upload with retries
        max_upload_attempts = 3
        upload_response = None
        
        for attempt in range(max_upload_attempts):
            try:
                logging.info(f"Upload attempt {attempt + 1}/{max_upload_attempts}")
                upload_response = upload_to_youtube(youtube, merged_path, title, description)
                if upload_response:
                    break
                time.sleep(60)  # Wait longer between attempts
            except Exception as e:
                logging.error(f"Upload attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_upload_attempts - 1:
                    logging.info("Waiting 60 seconds before next attempt...")
                    time.sleep(60)
                continue
        
        if not upload_response or not upload_response.get('id'):
            logging.error("Upload failed after all attempts")
            return
            
        video_id = upload_response['id']
        logging.info(f"Upload successful! Video ID: {video_id}")
        
        # Save link file in the tutorial folder
        cleanup_and_save_link(tutorial_path, video_id)
        logging.info(f"Video URL: https://www.youtube.com/watch?v={video_id}")
        
        # Only proceed with cleanup if we have a valid video ID and link file
        if os.path.exists(link_file):
            # Delete all folders containing videos
            if delete_video_folders(tutorial_path):
                logging.info("Successfully deleted all video folders")
            else:
                logging.error("Failed to delete some video folders")
                
            # Clean up merged video file
            try:
                os.remove(merged_path)
                logging.info(f"Cleaned up merged video file: {merged_path}")
            except Exception as e:
                logging.warning(f"Could not clean up merged video: {str(e)}")
        else:
            logging.warning("Link file not created. Skipping cleanup to be safe.")
            
    except Exception as e:
        logging.error(f"Error processing tutorial folder: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        # Clean up temporary folder if it exists
        if temp_folder and os.path.exists(temp_folder):
            try:
                shutil.rmtree(temp_folder)
            except:
                pass

def find_tutorial_folders(parent_path: str) -> List[str]:
    """Find all tutorial folders under the given parent path."""
    tutorial_folders = []
    try:
        logging.info(f"\nScanning for tutorial folders in: {parent_path}")
        
        # First, look for direct tutorial folders (level 4)
        if is_tutorial_folder(parent_path):
            # Check if this folder has videos
            has_videos = False
            for root, _, files in os.walk(parent_path):
                if any(f.lower().endswith(('.mp4', '.avi', '.mkv')) for f in files):
                    has_videos = True
                    break
            if has_videos:
                tutorial_folders.append(parent_path)
                logging.info(f"Found tutorial folder: {parent_path}")
        
        # Then look for subfolders that might be tutorial folders
        for root, dirs, _ in os.walk(parent_path):
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                
                # Skip if already processed
                if os.path.exists(os.path.join(full_path, "youtube link.txt")):
                    logging.info(f"Skipping already processed folder: {full_path}")
                    continue
                
                # Check if this is a tutorial folder
                if is_tutorial_folder(full_path):
                    # Check if this folder has videos
                    has_videos = False
                    for _, _, files in os.walk(full_path):
                        if any(f.lower().endswith(('.mp4', '.avi', '.mkv')) for f in files):
                            has_videos = True
                            break
                    
                    if has_videos:
                        tutorial_folders.append(full_path)
                        logging.info(f"Found tutorial folder: {full_path}")
                    else:
                        logging.info(f"Skipping folder with no videos: {full_path}")
        
        # Sort folders by path for consistent processing order
        tutorial_folders.sort()
        
        if tutorial_folders:
            logging.info(f"\nFound {len(tutorial_folders)} tutorial folders to process:")
            for folder in tutorial_folders:
                logging.info(f"  - {folder}")
        else:
            logging.warning("No tutorial folders with videos found")
            
        return tutorial_folders
        
    except Exception as e:
        logging.error(f"Error finding tutorial folders: {str(e)}")
        return []

def process_folder(parent_path: str, youtube: object):
    """Process all tutorial folders under the given parent path."""
    try:
        # Find all tutorial folders
        tutorial_folders = find_tutorial_folders(parent_path)
        
        if not tutorial_folders:
            logging.error("No tutorial folders with videos found")
            return
        
        # Process each tutorial folder
        total_folders = len(tutorial_folders)
        for i, tutorial_folder in enumerate(tutorial_folders, 1):
            logging.info(f"\nProcessing tutorial {i}/{total_folders}: {tutorial_folder}")
            
            # Check if already processed
            if os.path.exists(os.path.join(tutorial_folder, "youtube link.txt")):
                logging.info("Folder already processed (found youtube link.txt). Skipping.")
                continue
                
            try:
                process_tutorial_folder(tutorial_folder, youtube)
            except Exception as e:
                logging.error(f"Error processing tutorial folder {tutorial_folder}: {str(e)}")
                continue
            
    except Exception as e:
        logging.error(f"Error in process_folder: {str(e)}")

def get_script_directory():
    """Get the directory containing the script, works in both .py and .ipynb"""
    try:
        # For .py files
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # For Jupyter notebooks
        try:
            import nbformat
            from notebook import notebookapp
            import jupyter_client
            
            # Try to get the current notebook's path
            connection_file = jupyter_client.find_connection_file()
            kernel_id = os.path.basename(connection_file).split('-', 1)[1].split('.')[0]
            
            for srv in notebookapp.list_running_servers():
                for note in srv['notebook_dir']:
                    if kernel_id in note:
                        return os.path.dirname(note['path'])
                        
        except:
            # Fallback: Look in current working directory
            return os.getcwd()

def find_client_secrets():
    """Find client_secret.json in various possible locations"""
    possible_locations = [
        # Current directory
        os.path.join(os.getcwd(), "client_secret.json"),
        # Script directory (for .py files)
        os.path.join(get_script_directory(), "client_secret.json"),
        # One level up
        os.path.join(os.path.dirname(os.getcwd()), "client_secret.json"),
        # In 04.merge.and.upload directory
        os.path.join(os.getcwd(), "04.merge.and.upload", "client_secret.json"),
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            print(f"Found client_secret.json at: {location}")
            return location
            
    print("Searched in the following locations:")
    for location in possible_locations:
        print(f"- {location}")
    return None

def main():
    try:
        print("\n=== Starting YouTube Video Processor ===")
        
        # Find client secrets file
        client_secrets_file = find_client_secrets()
        if not client_secrets_file:
            print("Error: client_secret.json not found in any expected location")
            return
            
        # Authenticate with YouTube as the very first step
        youtube = authenticate_youtube(client_secrets_file)
        if not youtube:
            print("Failed to authenticate with YouTube. Please try again.")
            return
            
        # Now get folder path using GUI
        print("\nOpening folder selection window...")
        folder_path = get_folder_path_gui()
        if not folder_path:
            print("No path entered")
            return
            
        if not os.path.exists(folder_path):
            print("Path does not exist")
            return
            
        # Process folders with the authenticated service
        print(f"\nStarting to process folder: {folder_path}")
        process_folder(folder_path, youtube)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        logging.error(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
