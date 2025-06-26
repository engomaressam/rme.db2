#!/usr/bin/env python3
import os
import glob
import json
import time
import shutil
import logging
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

# Disable SSL verification globally
ssl._create_default_https_context = ssl._create_unverified_context

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

def get_authenticated_service(client_secrets_file: str):
    """Get authenticated YouTube service with proper token handling."""
    creds = None

    # Check if token file exists
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            logging.info("Loaded existing token")
        except Exception as e:
            logging.warning(f"Error loading token file: {e}")
            creds = None

    # If token is missing or invalid, refresh or re-authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logging.info("Token expired. Attempting to refresh...")
                creds.refresh(Request())
                logging.info("Token refreshed successfully!")
            except Exception as e:
                logging.warning(f"Token refresh failed: {e}")
                creds = None

        if not creds:
            logging.info("Opening browser for authentication...")
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the new token
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
            logging.info("Saved new token")

    return build("youtube", "v3", credentials=creds)

def get_structured_title(path: str) -> str:
    """Generate a structured title from the path levels after '05 Tutorials'."""
    try:
        parts = Path(path).parts
        # Find the index of "05 Tutorials"
        for i, part in enumerate(parts):
            if part == "05 Tutorials":
                # Take exactly 4 parts after "05 Tutorials"
                relevant_parts = parts[i+1:i+5]  # This will get: Google, Slides, Lynda, Google Slides Essential Training
                return " - ".join(relevant_parts)
        return os.path.basename(path)
    except Exception as e:
        logging.error(f"Error in get_structured_title: {str(e)}")
        return os.path.basename(path)

def is_tutorial_folder(path: str) -> bool:
    """
    Check if this is a level 4 tutorial folder (e.g. 'Google Slides Essential Training').
    Path structure: .../05 Tutorials/level1/level2/level3/level4
    """
    try:
        parts = Path(path).parts
        # Find the index of "05 Tutorials"
        for i, part in enumerate(parts):
            if part == "05 Tutorials":
                # Check if we have exactly 4 levels after "05 Tutorials"
                remaining_parts = parts[i+1:]
                return len(remaining_parts) == 4
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
    """Get video duration using ffprobe with better error handling."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", 
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", 
             video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        duration = float(result.stdout.strip())
        return duration
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

def merge_videos(video_files: List[Tuple[str, str]], output_path: str) -> bool:
    """Merge videos using ffmpeg with proper format settings."""
    try:
        # Create file list
        list_path = os.path.join(os.path.dirname(output_path), "file_list.txt")
        with open(list_path, 'w', encoding='utf-8') as f:
            for full_path, _ in video_files:
                f.write(f"file '{full_path}'\n")
        
        # Calculate expected duration
        expected_duration = sum(get_video_duration(full_path) or 0 for full_path, _ in video_files)
        if expected_duration == 0:
            logging.error("Could not calculate total duration of videos")
            return False
            
        logging.info(f"Expected duration: {expected_duration:.2f} seconds")
        
        # Merge videos with specific format settings
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264",     # Use H.264 codec
            "-preset", "fast",      # Faster encoding
            "-crf", "23",          # Good quality/size balance
            "-c:a", "aac",         # AAC audio codec
            "-b:a", "128k",        # Audio bitrate
            "-movflags", "+faststart",  # Web playback optimization
            output_path,
            "-y"  # Overwrite output file
        ]
        
        # Run ffmpeg
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Verify output duration
        output_duration = get_video_duration(output_path)
        if output_duration is None:
            logging.error("Could not verify output video duration")
            return False
            
        # Allow 5 second tolerance
        if abs(output_duration - expected_duration) > 5:
            logging.error(f"Output duration ({output_duration:.2f}s) doesn't match expected ({expected_duration:.2f}s)")
            return False
            
        logging.info(f"Merged video duration: {output_duration:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error merging videos: {str(e)}")
        return False
    finally:
        if os.path.exists(list_path):
            os.remove(list_path)

def upload_to_youtube(youtube, video_path: str, title: str, description: str) -> Optional[str]:
    """Upload video to YouTube with improved error handling."""
    try:
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

        insert_request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=MediaFileUpload(
                video_path, 
                chunksize=-1,  # Use default chunk size
                resumable=True
            )
        )

        response = None
        retries = 3
        while response is None and retries > 0:
            try:
                _, response = insert_request.next_chunk()
                if response:
                    logging.info("Upload completed successfully")
                    return response["id"]
            except Exception as e:
                retries -= 1
                if retries == 0:
                    raise
                logging.warning(f"Upload chunk failed, retrying... ({str(e)})")
                time.sleep(5)  # Wait before retry

    except Exception as e:
        logging.error(f"Error uploading to YouTube: {str(e)}")
        return None

def cleanup_and_save_link(folder_path: str, video_id: str, title: str):
    """Save YouTube link and clean up files."""
    link_file = os.path.join(folder_path, "youtube link.txt")
    with open(link_file, 'w') as f:
        f.write(f"Title: {title}\n")
        f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
        f.write(f"Uploaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def process_tutorial_folder(tutorial_path: str, client_secrets_file: str):
    """Process a single tutorial folder containing videos."""
    try:
        logging.info(f"\nProcessing tutorial folder: {tutorial_path}")
        
        # Verify this is a level 4 tutorial folder
        if not is_tutorial_folder(tutorial_path):
            logging.error(f"Not a tutorial folder (level 4): {tutorial_path}")
            return
        
        # Get videos from this folder and all subfolders
        video_files = collect_videos(tutorial_path)
        if not video_files:
            logging.error(f"No video files found in {tutorial_path}")
            return
            
        logging.info(f"Found {len(video_files)} videos:")
        for _, rel_path in video_files:
            logging.info(f"  - {rel_path}")
        
        # Get title (will be "Google - Slides - Lynda - Google Slides Essential Training")
        title = get_structured_title(tutorial_path)
        logging.info(f"Generated title: {title}")
        
        # Create merged file in the tutorial folder
        merged_filename = f"{os.path.basename(tutorial_path)}_merged.mp4"
        merged_path = os.path.join(tutorial_path, merged_filename)
        
        logging.info("Starting video merge...")
        if not merge_videos(video_files, merged_path):
            logging.error("Failed to merge videos")
            return
            
        logging.info("Videos merged successfully")
        
        # Generate timestamps
        timestamps = generate_timestamps(video_files)
        description = "Tutorial Contents:\n\n" + timestamps
        
        # Get authenticated service (with proper token handling)
        try:
            youtube = get_authenticated_service(client_secrets_file)
            logging.info("YouTube authentication successful")
        except Exception as e:
            logging.error(f"Failed to authenticate with YouTube: {str(e)}")
            return
            
        # Upload with retries
        max_upload_attempts = 3
        video_id = None
        
        for attempt in range(max_upload_attempts):
            try:
                logging.info(f"Upload attempt {attempt + 1}/{max_upload_attempts}")
                video_id = upload_to_youtube(youtube, merged_path, title, description)
                if video_id:
                    break
                time.sleep(5)  # Wait between attempts
            except Exception as e:
                logging.error(f"Upload attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_upload_attempts - 1:
                    time.sleep(10)  # Longer wait before next attempt
                continue
        
        if video_id:
            logging.info(f"Upload successful! Video ID: {video_id}")
            # Save link file in the tutorial folder
            cleanup_and_save_link(tutorial_path, video_id, title)
            logging.info(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            
            # Clean up merged video file after successful upload
            try:
                os.remove(merged_path)
                logging.info(f"Cleaned up merged video file: {merged_path}")
            except Exception as e:
                logging.warning(f"Could not clean up merged video: {str(e)}")
        else:
            logging.error("Upload failed after all attempts")
            
    except Exception as e:
        logging.error(f"Error processing tutorial folder: {str(e)}")

def find_tutorial_folders(parent_path: str) -> List[str]:
    """Find all level 4 tutorial folders under the given parent path."""
    tutorial_folders = []
    try:
        # Walk through all subdirectories
        for root, _, files in os.walk(parent_path):
            # Check if this is a tutorial folder (level 4)
            if is_tutorial_folder(root):
                # Check if this folder has videos (directly or in subfolders)
                has_videos = False
                for _, _, fs in os.walk(root):
                    if any(f.lower().endswith(('.mp4', '.avi', '.mkv')) for f in fs):
                        has_videos = True
                        break
                if has_videos:
                    tutorial_folders.append(root)
    except Exception as e:
        logging.error(f"Error finding tutorial folders: {str(e)}")
    return sorted(tutorial_folders)

def process_folder(parent_path: str, client_secrets_file: str):
    """Process all tutorial folders under the given parent path."""
    try:
        logging.info(f"Scanning for tutorial folders in: {parent_path}")
        
        # Find all tutorial folders
        tutorial_folders = find_tutorial_folders(parent_path)
        
        if not tutorial_folders:
            logging.error("No tutorial folders with videos found")
            return
            
        logging.info(f"Found {len(tutorial_folders)} tutorial folders")
        
        # Process each tutorial folder
        for tutorial_folder in tutorial_folders:
            logging.info(f"\nProcessing tutorial: {tutorial_folder}")
            process_tutorial_folder(tutorial_folder, client_secrets_file)
            
    except Exception as e:
        logging.error(f"Error in process_folder: {str(e)}")

def main():
    try:
        # Get folder path using GUI
        folder_path = get_folder_path_gui()
        if not folder_path:
            print("No path entered")
            return
            
        if not os.path.exists(folder_path):
            print("Path does not exist")
            return
            
        # Get client secrets file
        client_secrets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_secret.json")
        if not os.path.exists(client_secrets_file):
            print("Client secrets file not found")
            return
            
        process_folder(folder_path, client_secrets_file)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
