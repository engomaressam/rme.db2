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
import string
import re
import tempfile

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

# Add global lists to track failed conversions and merges
FAILED_CONVERSIONS = []
FAILED_MERGES = []
SKIPPED_FILES = []

def get_authenticated_service(client_secrets_file: str):
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
    try:
        path = Path(input_path)
        parts = list(path.parts)[-4:]
        return ' - '.join(parts)
    except Exception as e:
        logging.error(f"Error in get_structured_title: {str(e)}")
        return os.path.basename(input_path)

def collect_videos(folder_path: str) -> List[Tuple[str, str]]:
    video_files = []
    extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.flv', '*.rmvb')
    for root, dirs, files in os.walk(folder_path):
        # Skip any 'converted_videos' subfolder
        if 'converted_videos' in root.split(os.sep):
            continue
        for ext in extensions:
            for file in glob.glob(os.path.join(root, ext)):
                rel_path = os.path.relpath(file, folder_path)
                video_files.append((file, rel_path))
    return sorted(video_files)

def merge_videos(video_files: List[Tuple[str, str]], output_path: str) -> bool:
    try:
        valid_files = []
        file_list_path = None
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.txt', encoding='utf-8') as tmpfile:
            for full_path, _ in video_files:
                if not os.path.exists(full_path):
                    logging.error(f"File does not exist, skipping: {full_path}")
                    SKIPPED_FILES.append(full_path)
                    continue
                if get_video_duration(full_path) <= 0:
                    logging.error(f"File is not a valid video or is corrupt, skipping: {full_path}")
                    SKIPPED_FILES.append(full_path)
                    continue
                abs_path = os.path.abspath(full_path).replace('\\', '/')
                abs_path = abs_path.replace("'", "'\\''")
                tmpfile.write(f"file '{abs_path}'\n")
                valid_files.append(full_path)
            file_list_path = tmpfile.name
        if not valid_files:
            logging.error(f"No valid files to merge for output: {output_path}")
            FAILED_MERGES.append(output_path)
            return False
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            output_path,
            '-y'
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if process.returncode != 0:
            logging.error(f"Error merging videos: {process.stderr}")
            FAILED_MERGES.append(output_path)
            # Log the list of files in this batch for debugging
            logging.error(f"Files in failed batch merge for {output_path}: {valid_files}")
        return process.returncode == 0
    except Exception as e:
        logging.error(f"Error merging videos: {str(e)}")
        FAILED_MERGES.append(output_path)
        return False
    finally:
        if 'file_list_path' in locals() and file_list_path and os.path.exists(file_list_path):
            os.remove(file_list_path)

def convert_video(input_path: str, output_path: str) -> bool:
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vf', 'scale=-2:720',
            '-r', '30',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            output_path
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if process.returncode != 0:
            logging.error(f"Error converting {input_path}: {process.stderr}")
            FAILED_CONVERSIONS.append(input_path)
        return process.returncode == 0
    except Exception as e:
        logging.error(f"Exception converting {input_path}: {str(e)}")
        FAILED_CONVERSIONS.append(input_path)
        return False

def convert_all_videos(video_files: List[Tuple[str, str]], converted_dir: str) -> List[Tuple[str, str]]:
    os.makedirs(converted_dir, exist_ok=True)
    converted_files = []
    for full_path, rel_path in video_files:
        base_name = os.path.splitext(rel_path)[0] + '.mp4'
        out_path = os.path.join(converted_dir, base_name)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        if convert_video(full_path, out_path):
            converted_files.append((out_path, rel_path))
        else:
            logging.error(f"Failed to convert {full_path}")
    return converted_files

def sanitize_description(desc: str) -> str:
    """Remove non-printable characters and truncate to 5000 chars for YouTube description."""
    desc = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uD7FF\uE000-\uFFFD]', '', desc)
    return desc[:5000]

def generate_timestamps(video_files: List[Tuple[str, str]]) -> str:
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

def cleanup_and_save_link(folder_path: str, video_id: str, title: str, merged_path: str, converted_dir: Optional[str] = None, delete_all: bool = False, original_files: Optional[list] = None):
    link_file = os.path.join(folder_path, "youtube link.txt")
    with open(link_file, 'w') as f:
        f.write(f"Title: {title}\n")
        f.write(f"URL: https://www.youtube.com/watch?v={video_id}\n")
        f.write(f"Uploaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Delete merged video
    if os.path.exists(merged_path):
        try:
            os.remove(merged_path)
        except Exception as e:
            logging.error(f"Failed to delete merged video: {merged_path}: {str(e)}")
    # Delete all video files in the top-level tutorial folder (except the link file)
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if item_path == link_file:
            continue
        if os.path.isfile(item_path) and item_path.lower().endswith((".mp4", ".flv", ".avi", ".mkv", ".mov", ".rmvb")):
            try:
                os.remove(item_path)
            except Exception as e:
                logging.error(f"Failed to delete video file: {item_path}: {str(e)}")
        elif os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
            except Exception as e:
                logging.error(f"Failed to delete folder: {item_path}: {str(e)}")
    # Remove converted_videos folder if present
    if converted_dir and os.path.exists(converted_dir):
        try:
            shutil.rmtree(converted_dir)
        except Exception as e:
            logging.error(f"Failed to delete converted_videos: {converted_dir}: {str(e)}")
    # If delete_all is True, remove all original video files in all subfolders as well
    if delete_all:
        # Remove all original video files (not just converted) in all subfolders
        if original_files:
            for orig_file in original_files:
                if os.path.exists(orig_file):
                    try:
                        os.remove(orig_file)
                    except Exception as e:
                        logging.error(f"Failed to delete original video file: {orig_file}: {str(e)}")
        else:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith((".mp4", ".avi", ".mkv", ".mov", ".flv", ".rmvb")):
                        try:
                            os.remove(os.path.join(root, file))
                        except Exception as e:
                            logging.error(f"Failed to delete {file}: {str(e)}")
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if dir_path != converted_dir and os.path.exists(dir_path):
                        try:
                            shutil.rmtree(dir_path)
                        except Exception as e:
                            logging.error(f"Failed to delete directory {dir_path}: {str(e)}")

def get_video_duration(video_path: str) -> float:
    try:
        probe = ffmpeg.probe(video_path)
        for stream in probe['streams']:
            if 'duration' in stream:
                return float(stream['duration'])
        return float(probe['format']['duration'])
    except Exception as e:
        logging.error(f"Error getting duration for {video_path}: {str(e)}")
        SKIPPED_FILES.append(video_path)
        return 0.0

def sanitize_title(title):
    # Remove non-printable characters
    printable = set(string.printable)
    title = ''.join(filter(lambda x: x in printable, title))
    # Truncate to 100 chars
    return title[:100]

def split_videos_by_duration(video_files: List[Tuple[str, str]], durations: List[float], max_duration: float = 41400) -> List[Tuple[List[Tuple[str, str]], List[float], str]]:
    """
    Splits the list of video files into as many parts as needed so that each part's total duration is <= max_duration (default 11.5 hours).
    Returns a list of tuples: (split_files, split_durations, suffix)
    """
    splits = []
    part = []
    part_durations = []
    part_total = 0.0
    part_idx = 1
    for (file, rel), dur in zip(video_files, durations):
        if part_total + dur > max_duration and part:
            splits.append((part, part_durations, f"{part_idx:02d}"))
            part = []
            part_durations = []
            part_total = 0.0
            part_idx += 1
        part.append((file, rel))
        part_durations.append(dur)
        part_total += dur
    if part:
        splits.append((part, part_durations, f"{part_idx:02d}"))
    return splits

def batch_merge(video_files: List[Tuple[str, str]], batch_size: int, temp_dir: str) -> List[Tuple[str, str]]:
    os.makedirs(temp_dir, exist_ok=True)
    batch_files = []
    for i in range(0, len(video_files), batch_size):
        batch = video_files[i:i+batch_size]
        batch_path = os.path.join(temp_dir, f"batch_{i//batch_size:03d}.mp4")
        if merge_videos(batch, batch_path):
            batch_files.append((batch_path, f"batch_{i//batch_size:03d}.mp4"))
        else:
            logging.error(f"Batch merge failed for {batch_path}")
            FAILED_MERGES.append(batch_path)
    return batch_files

def process_folder(folder_path: str, client_secrets_file: str):
    try:
        if os.path.basename(folder_path) == 'converted_videos':
            logging.error("Refusing to process a folder named 'converted_videos' to avoid recursion.")
            return
        logging.info(f"Processing folder: {folder_path}")
        # REV 16: Check for existing merged video and youtube link
        link_file = os.path.join(folder_path, "youtube link.txt")
        merged_videos = [f for f in os.listdir(folder_path) if f.endswith('.mp4') and '_merged' in f]
        if os.path.exists(link_file):
            logging.info(f"YouTube link file already exists for {folder_path}, skipping upload.")
            return
        if merged_videos:
            # If multiple merged videos, upload each (for split parts)
            merged_videos = sorted(merged_videos)
            for merged_name in merged_videos:
                merged_path = os.path.join(folder_path, merged_name)
                if not os.path.exists(merged_path):
                    logging.warning(f"Merged video file not found: {merged_path}, skipping upload.")
                    continue
                # Use structured title (last 4 folders separated by dashes)
                title = get_structured_title(folder_path)
                if not title or not title.strip():
                    title = os.path.splitext(merged_name)[0]
                # If there are multiple merged videos, add suffix
                if len(merged_videos) > 1:
                    suffix = os.path.splitext(merged_name)[0].replace(os.path.basename(folder_path), '').replace('_merged', '').strip()
                    if suffix:
                        title = f"{title} {suffix}"
                title = sanitize_title(title)
                # Try to generate timestamps from original videos if they exist
                video_files = collect_videos(folder_path)
                if video_files:
                    timestamps = generate_timestamps(video_files)
                    description = sanitize_description("Tutorial Contents:\n\n" + timestamps)
                else:
                    description = sanitize_description("Tutorial video upload.")
                youtube = get_authenticated_service(client_secrets_file)
                video_id = upload_to_youtube(youtube, merged_path, title, description)
                if video_id:
                    logging.info(f"Upload successful! Video ID: {video_id}")
                    # Clean up and save link as in normal flow
                    cleanup_and_save_link(folder_path, video_id, title, merged_path)
                    logging.info(f"Video URL: https://www.youtube.com/watch?v={video_id}")
                else:
                    logging.error(f"Upload failed for merged video: {merged_name}")
            return
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
            if d == 0.0:
                logging.error(f"Skipping unreadable/corrupt file: {file}")
                continue
            durations.append(d)
            total_duration += d
        if not durations:
            logging.error("No valid video files after filtering corrupt/unreadable files.")
            return
        logging.info(f"Total duration: {total_duration/3600:.2f} hours")
        title_base = get_structured_title(folder_path)
        if not title_base or not title_base.strip():
            title_base = os.path.basename(folder_path)
        logging.info(f"Generated title: {title_base}")
        # Split into as many parts as needed so each is <= 11.5 hours
        if total_duration > 41400 and len(video_files) > 1:
            splits = split_videos_by_duration(video_files, durations, max_duration=41400)
        else:
            splits = [(video_files, durations, None)]
        for idx, (split_files, split_durations, suffix) in enumerate(splits):
            merged_path = os.path.join(folder_path, f"{os.path.basename(folder_path)}_merged{suffix or ''}.mp4")
            # Try direct merge first
            # If too many files, do batch merge
            if len(split_files) > 100:
                temp_batch_dir = os.path.join(folder_path, 'batch_merge_temp')
                batch_files = batch_merge(split_files, 100, temp_batch_dir)
                merged_ok = merge_videos(batch_files, merged_path)
                # Clean up batch files
                for f, _ in batch_files:
                    if os.path.exists(f):
                        os.remove(f)
                if os.path.exists(temp_batch_dir):
                    shutil.rmtree(temp_batch_dir)
            else:
                merged_ok = merge_videos(split_files, merged_path)
            converted_dir = None
            use_conversion = False
            merged_duration = get_video_duration(merged_path) if merged_ok else 0
            sum_original = sum([get_video_duration(f) for f, _ in split_files])
            # Sanity check: merged duration should be within 10% of sum of originals
            if merged_ok and sum_original > 0:
                diff_ratio = abs(merged_duration - sum_original) / sum_original
                if diff_ratio > 0.10:
                    logging.warning(f"Merged video duration {merged_duration/3600:.2f}h differs by more than 10% from originals ({sum_original/3600:.2f}h). Will try safe conversion path.")
                    merged_ok = False
                    use_conversion = True
                    if os.path.exists(merged_path):
                        os.remove(merged_path)
            # If merged video is more than 3x the sum of originals and >8 hours, treat as failed
            if merged_ok and merged_duration > max(8*3600, 3*sum_original):
                logging.warning(f"Merged video duration {merged_duration/3600:.2f}h is much larger than originals ({sum_original/3600:.2f}h). Will try safe conversion path.")
                merged_ok = False
                use_conversion = True
                if os.path.exists(merged_path):
                    os.remove(merged_path)
            if not merged_ok:
                logging.warning(f"Direct merge failed or unsafe for part {suffix or 'single'}. Attempting conversion.")
                # Convert all to mp4 in a subfolder
                if os.path.basename(folder_path) == 'converted_videos':
                    converted_dir = os.path.join(folder_path, 'converted_videos_2')
                else:
                    converted_dir = os.path.join(folder_path, 'converted_videos')
                converted_files = convert_all_videos(split_files, converted_dir)
                if not converted_files:
                    logging.error(f"Conversion failed for all videos in part {suffix or 'single'}.")
                    continue
                # If too many files, do batch merge after conversion
                if len(converted_files) > 100:
                    temp_batch_dir = os.path.join(folder_path, 'batch_merge_temp')
                    batch_files = batch_merge(converted_files, 100, temp_batch_dir)
                    merged_ok = merge_videos(batch_files, merged_path)
                    for f, _ in batch_files:
                        if os.path.exists(f):
                            os.remove(f)
                    if os.path.exists(temp_batch_dir):
                        shutil.rmtree(temp_batch_dir)
                else:
                    merged_ok = merge_videos([(f, r) for f, r in converted_files], merged_path)
                if not merged_ok:
                    logging.error(f"Failed to merge even after conversion for part {suffix or 'single'}")
                    continue
                split_files = [(f, r) for f, r in converted_files]  # For timestamps
                # Recalculate durations after conversion
                merged_duration = get_video_duration(merged_path)
                sum_original = sum([get_video_duration(f) for f, _ in split_files])
                # Sanity check again after conversion
                if sum_original > 0:
                    diff_ratio = abs(merged_duration - sum_original) / sum_original
                    if diff_ratio > 0.10:
                        logging.error(f"Merged video after conversion still differs by more than 10% from originals. Skipping part {suffix or 'single'}.")
                        if os.path.exists(merged_path):
                            os.remove(merged_path)
                        continue
            logging.info(f"Videos merged successfully for part {suffix or 'single'}")
            # Check merged video duration (should be < 11.5h, but check anyway)
            duration = get_video_duration(merged_path)
            if duration > 41400:
                logging.warning(f"Merged video is too long ({duration/3600:.2f} hours). Skipping upload and cleanup for: {merged_path}")
                continue
            timestamps = generate_timestamps(split_files)
            description = sanitize_description("Tutorial Contents:\n\n" + timestamps)
            title = title_base if not suffix else f"{title_base} {suffix}"
            title = title.strip()
            if not title:
                title = os.path.basename(folder_path)
            title = sanitize_title(title)
            logging.info(f"Uploading with title: '{title}' (length: {len(title)})")
            if not title or not title.strip():
                title = os.path.basename(folder_path)[:100]
                logging.warning(f"Title was empty after sanitization, using fallback: '{title}'")
            youtube = get_authenticated_service(client_secrets_file)
            video_id = upload_to_youtube(youtube, merged_path, title, description)
            if video_id:
                logging.info(f"Upload successful! Video ID: {video_id}")
                # Pass original files to cleanup if conversion was used
                orig_files = [f for f, _ in split_files] if use_conversion else None
                cleanup_and_save_link(folder_path, video_id, title, merged_path, converted_dir, delete_all=use_conversion, original_files=orig_files)
                logging.info(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            else:
                logging.error(f"Upload failed for part {suffix or 'single'}")
        # Write skipped files summary
        if SKIPPED_FILES:
            skipped_path = os.path.join(folder_path, 'skipped_files.txt')
            with open(skipped_path, 'w', encoding='utf-8') as f:
                for path in SKIPPED_FILES:
                    f.write(path + '\n')
            logging.info(f"Skipped files written to {skipped_path}")
        # Log failed conversions and merges
        if FAILED_CONVERSIONS:
            logging.error(f"Failed conversions: {FAILED_CONVERSIONS}")
        if FAILED_MERGES:
            logging.error(f"Failed merges: {FAILED_MERGES}")
    except Exception as e:
        logging.error(f"Error processing folder: {str(e)}")

def find_tutorial_folders_2_levels_down(root_folder: str) -> list:
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
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    client_secrets_file = os.path.join(script_dir, "client_secret.json")
    if not os.path.exists(client_secrets_file):
        print("Client secrets file not found")
        return
    youtube = get_authenticated_service(client_secrets_file)
    print("YouTube authentication complete.")
    root_folder = select_folder_dialog("Select the software folder (2 levels above tutorial)")
    if not root_folder or not os.path.exists(root_folder):
        print("Path does not exist or was not selected.")
        return
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