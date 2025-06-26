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
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import re

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

def get_authenticated_service(client_secrets_file: str):
    """Get authenticated YouTube service."""
    creds = None
    
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
            
    return build('youtube', 'v3', credentials=creds)

def get_structured_title(input_path: str) -> str:
    """Return the last 4 parts of the path, joined by ' - ', in order from outermost to innermost."""
    try:
        path = Path(input_path)
        parts = list(path.parts)[-4:]
        return ' - '.join(parts)
    except Exception as e:
        logging.error(f"Error in get_structured_title: {str(e)}")
        return os.path.basename(input_path)

def collect_videos(folder_path: str) -> List[Tuple[str, str]]:
    """Collect all videos from folder and subfolders."""
    video_files = []
    for ext in ('*.mp4', '*.avi', '*.mkv'):
        for file_path in glob.glob(os.path.join(folder_path, "**", ext), recursive=True):
            rel_path = os.path.relpath(file_path, folder_path)
            video_files.append((file_path, rel_path))
    return sorted(video_files)

def merge_videos(video_files: List[Tuple[str, str]], output_path: str) -> bool:
    """Merge videos using ffmpeg."""
    try:
        with open('file_list.txt', 'w', encoding='utf-8') as f:
            for full_path, _ in video_files:
                f.write(f"file '{full_path}'\n")
        
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', 'file_list.txt',
            '-c', 'copy',
            output_path,
            '-y'
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        return process.returncode == 0
        
    except Exception as e:
        logging.error(f"Error merging videos: {str(e)}")
        return False
    finally:
        if os.path.exists('file_list.txt'):
            os.remove('file_list.txt')

def sanitize_description(desc: str) -> str:
    """Remove non-printable characters and truncate to 5000 chars for YouTube description."""
    # Remove non-printable characters (except common whitespace)
    desc = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uE000-\uFFFD]', '', desc)
    # Truncate to 5000 chars
    return desc[:5000]

def generate_timestamps(video_files: List[Tuple[str, str]]) -> str:
    """Generate timestamps for video description with improved ffprobe error handling."""
    timestamps = []
    current_time = 0
    for full_path, rel_path in video_files:
        hours = int(current_time // 3600)
        minutes = int((current_time % 3600) // 60)
        seconds = int(current_time % 60)
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        video_name = os.path.splitext(os.path.basename(rel_path))[0]
        timestamps.append(f"{timestamp} - {video_name}")
        try:
            # Use ffprobe via subprocess for better error capture
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', full_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                current_time += duration
            else:
                logging.error(f"ffprobe error for {full_path}: {result.stderr.strip()}")
                current_time += 0
        except Exception as e:
            logging.error(f"Exception running ffprobe for {full_path}: {str(e)}")
            current_time += 0
    return '\n'.join(timestamps)

def upload_to_youtube(youtube, video_path: str, title: str, description: str) -> Optional[str]:
    """Upload video to YouTube."""
    try:
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': ['tutorial', 'education']
            },
            'status': {
                'privacyStatus': 'private',
                'selfDeclaredMadeForKids': False
            }
        }

        insert_request = youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=MediaFileUpload(video_path, chunksize=-1, resumable=True)
        )

        response = None
        while response is None:
            status, response = insert_request.next_chunk()
            if status:
                logging.info(f"Upload {int(status.progress() * 100)}% complete")

        return response['id']

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

def process_folder(folder_path: str, client_secrets_file: str):
    """Process a folder of videos."""
    try:
        logging.info(f"Processing folder: {folder_path}")
        # Get videos
        video_files = collect_videos(folder_path)
        if not video_files:
            logging.error("No video files found")
            return
        logging.info(f"Found {len(video_files)} videos")
        # Get title
        title = get_structured_title(folder_path)
        logging.info(f"Generated title: {title}")
        # Merge videos
        merged_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_merged.mp4")
        if not merge_videos(video_files, merged_path):
            logging.error("Failed to merge videos")
            return
        logging.info("Videos merged successfully")
        # Generate timestamps
        timestamps = generate_timestamps(video_files)
        description = sanitize_description("Tutorial Contents:\n\n" + timestamps)
        # Upload to YouTube
        youtube = get_authenticated_service(client_secrets_file)
        video_id = upload_to_youtube(youtube, merged_path, title, description)
        if video_id:
            logging.info(f"Upload successful! Video ID: {video_id}")
            cleanup_and_save_link(folder_path, video_id, title)
            logging.info(f"Video URL: https://www.youtube.com/watch?v={video_id}")
        else:
            logging.error("Upload failed")
    except Exception as e:
        logging.error(f"Error processing folder: {str(e)}")

def find_leaf_folders(root_folder: str) -> list:
    """Find all leaf (tutorial) folders under the root_folder (folders with video files, no subfolders with video files)."""
    leaf_folders = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Check if this folder has video files
        has_videos = any(f.lower().endswith((".mp4", ".avi", ".mkv")) for f in filenames)
        # Check if any subfolder has video files
        subfolders_with_videos = False
        for d in dirnames:
            subfolder = os.path.join(dirpath, d)
            for _, _, subfiles in os.walk(subfolder):
                if any(f.lower().endswith((".mp4", ".avi", ".mkv")) for f in subfiles):
                    subfolders_with_videos = True
                    break
            if subfolders_with_videos:
                break
        if has_videos and not subfolders_with_videos:
            leaf_folders.append(dirpath)
    return leaf_folders

def main():
    # Get root folder path
    root_folder = input("Enter root folder path: ")
    if not os.path.exists(root_folder):
        print("Path does not exist")
        return
    # Get client secrets file
    client_secrets_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_secret.json")
    if not os.path.exists(client_secrets_file):
        print("Client secrets file not found")
        return
    # Find all leaf (tutorial) folders
    leaf_folders = find_leaf_folders(root_folder)
    if not leaf_folders:
        print("No tutorial folders found.")
        return
    print(f"Found {len(leaf_folders)} tutorial folders.")
    for folder in leaf_folders:
        print(f"Processing: {folder}")
        process_folder(folder, client_secrets_file)

if __name__ == "__main__":
    main()
