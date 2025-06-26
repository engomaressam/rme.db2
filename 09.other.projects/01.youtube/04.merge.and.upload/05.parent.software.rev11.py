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
import tkinter as tk
from tkinter import filedialog
import ssl

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

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['PYTHONHTTPSVERIFY'] = '0'

ssl._create_default_https_context = ssl._create_unverified_context

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

def generate_timestamps(video_files: List[Tuple[str, str]]) -> str:
    """Generate timestamps for video description."""
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
            probe = ffmpeg.probe(full_path)
            duration = float(probe['streams'][0]['duration'])
            current_time += duration
        except Exception as e:
            logging.error(f"Error getting duration for {rel_path}: {str(e)}")
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
                'privacyStatus': 'unlisted',
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

def cleanup_and_save_link(folder_path: str, video_id: str, title: str, merged_path: str):
    """Save YouTube link, delete merged video and all original videos/folders, leaving only the link file."""
    link_file = os.path.join(folder_path, "youtube link.txt")
    with open(link_file, 'w') as f:
        f.write(f"Title: {title}\n")
        f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
        f.write(f"Uploaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Delete merged video
    if os.path.exists(merged_path):
        os.remove(merged_path)
    # Delete all subfolders and video files in the tutorial folder, except the link file
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if item_path == link_file:
            continue
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif item_path.endswith(('.mp4', '.avi', '.mkv')):
            os.remove(item_path)

def get_video_duration(video_path: str) -> float:
    """Return duration of video in seconds using ffmpeg.probe."""
    try:
        probe = ffmpeg.probe(video_path)
        # Find the first video stream with a duration
        for stream in probe['streams']:
            if 'duration' in stream:
                return float(stream['duration'])
        # Fallback to format duration
        return float(probe['format']['duration'])
    except Exception as e:
        logging.error(f"Error getting duration for {video_path}: {str(e)}")
        return 0.0

def process_folder(folder_path: str, client_secrets_file: str):
    """Process a folder of videos. If total duration > 10 hours, split and upload as two videos."""
    try:
        logging.info(f"Processing folder: {folder_path}")
        video_files = collect_videos(folder_path)
        if not video_files:
            logging.error("No video files found")
            return
        logging.info(f"Found {len(video_files)} videos")
        # Calculate total duration
        total_duration = 0
        durations = []
        for file, _ in video_files:
            d = get_video_duration(file)
            durations.append(d)
            total_duration += d
        logging.info(f"Total duration: {total_duration/3600:.2f} hours")
        title_base = get_structured_title(folder_path)
        logging.info(f"Generated title: {title_base}")
        # If total duration > 10 hours, split
        if total_duration > 36000 and len(video_files) > 1:
            mid = len(video_files) // 2
            splits = [(video_files[:mid], durations[:mid], '01'), (video_files[mid:], durations[mid:], '02')]
        else:
            splits = [(video_files, durations, None)]
        for idx, (split_files, split_durations, suffix) in enumerate(splits):
            merged_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_merged{suffix or ''}.mp4")
            if not merge_videos(split_files, merged_path):
                logging.error(f"Failed to merge videos for part {suffix or 'single'}")
                continue
            logging.info(f"Videos merged successfully for part {suffix or 'single'}")
            # Check merged video duration (should be < 12h, but check anyway)
            duration = get_video_duration(merged_path)
            if duration > 43200:
                logging.warning(f"Merged video is too long ({duration/3600:.2f} hours). Skipping upload and cleanup for: {merged_path}")
                continue
            timestamps = generate_timestamps(split_files)
            description = "Tutorial Contents:\n\n" + timestamps
            title = title_base if not suffix else f"{title_base} {suffix}"
            youtube = get_authenticated_service(client_secrets_file)
            video_id = upload_to_youtube(youtube, merged_path, title, description)
            if video_id:
                logging.info(f"Upload successful! Video ID: {video_id}")
                cleanup_and_save_link(folder_path, video_id, title, merged_path)
                logging.info(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            else:
                logging.error(f"Upload failed for part {suffix or 'single'}")
    except Exception as e:
        logging.error(f"Error processing folder: {str(e)}")

def find_tutorial_folders_2_levels_down(root_folder: str) -> list:
    """Find all folders exactly 2 levels below the root_folder."""
    tutorial_folders = []
    for company in os.listdir(root_folder):
        company_path = os.path.join(root_folder, company)
        if os.path.isdir(company_path):
            for tutorial in os.listdir(company_path):
                tutorial_path = os.path.join(company_path, tutorial)
                if os.path.isdir(tutorial_path):
                    tutorial_folders.append(tutorial_path)
    return tutorial_folders

def select_folder_dialog(title="Select the software folder"):
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_selected

def main():
    # Get client secrets file
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    client_secrets_file = os.path.join(script_dir, "client_secret.json")
    if not os.path.exists(client_secrets_file):
        print("Client secrets file not found")
        return
    # Authenticate with YouTube first
    youtube = get_authenticated_service(client_secrets_file)
    print("YouTube authentication complete.")
    # Show folder selection dialog
    root_folder = select_folder_dialog("Select the software folder (2 levels above tutorial)")
    if not root_folder or not os.path.exists(root_folder):
        print("Path does not exist or was not selected.")
        return
    # Find all tutorial folders exactly 2 levels down
    tutorial_folders = find_tutorial_folders_2_levels_down(root_folder)
    if not tutorial_folders:
        print("No tutorial folders found.")
        return
    print(f"Found {len(tutorial_folders)} tutorial folders.")
    for folder in tutorial_folders:
        print(f"Processing: {folder}")
        process_folder(folder, client_secrets_file)

if __name__ == "__main__":
    main() 