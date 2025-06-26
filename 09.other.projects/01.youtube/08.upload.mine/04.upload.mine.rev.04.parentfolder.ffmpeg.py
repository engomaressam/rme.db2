import os
import pickle
import tkinter as tk
from tkinter import filedialog
import subprocess
import shlex
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
CLIENT_SECRETS_FILE = 'client_secret.emc.json'
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv']
MERGED_OUTPUT_DIR = 'D:/merged'
PROCESSED_JSON = os.path.join(MERGED_OUTPUT_DIR, 'processed_folders.json')

os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

def get_parent_directory():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select Parent Directory to Scan for Video Folders")
    root.destroy()
    return directory

def find_video_folders(parent_dir):
    video_folders = []
    for root, dirs, files in os.walk(parent_dir):
        if any(file.lower().endswith(tuple(VIDEO_EXTENSIONS)) for file in files):
            video_folders.append(root)
    return video_folders

def find_videos_by_resolution(directory):
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, file))
    # Group by resolution
    resolution_groups = {}
    for vf in video_files:
        try:
            result = subprocess.run([
                'python', os.path.join(os.path.dirname(__file__), 'ffprobe_resolution.py'), vf
            ], capture_output=True, text=True)
            if result.returncode == 0:
                res = result.stdout.strip()
                print(f"[RESOLUTION] {vf} -> {res}")
            else:
                res = 'unknown'
                print(f"[RESOLUTION-ERROR] {vf} -> unknown (ffprobe error: {result.stderr.strip()})")
        except Exception as e:
            res = 'unknown'
            print(f"[RESOLUTION-EXCEPTION] {vf} -> unknown (Exception: {e})")
        if res not in resolution_groups:
            resolution_groups[res] = []
        resolution_groups[res].append(vf)
    return resolution_groups

def merge_videos_ffmpeg(video_files, output_path):
    filelist_path = 'ffmpeg_filelist.txt'
    with open(filelist_path, 'w', encoding='utf-8') as f:
        for vf in video_files:
            f.write(f"file '{vf.replace('\\', '/').replace("'", "'\\''")}'\n")
    cmd = f"ffmpeg -f concat -safe 0 -i {filelist_path} -c copy \"{output_path}\""
    print(f"Running: {cmd}")
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        print('ffmpeg error:', result.stderr)
        raise RuntimeError('ffmpeg merge failed')
    os.remove(filelist_path)

def authenticate_youtube():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('youtube', 'v3', credentials=creds)

def get_last_4_folders(path):
    parts = os.path.normpath(path).split(os.sep)
    return ' - '.join(parts[-4:]) if len(parts) >= 4 else os.path.basename(path)

def upload_video(youtube, file_path, title):
    body = {
        'snippet': {
            'title': title,
            'description': 'Merged video uploaded by automated script',
            'tags': [],
            'categoryId': '22'
        },
        'status': {
            'privacyStatus': 'private'
        }
    }
    media = MediaFileUpload(file_path, resumable=True)
    request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )
    response = None
    try:
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Uploading {file_path}: {int(status.progress() * 100)}% done.")
        print(f"Upload Complete: {file_path}")
        # Compose YouTube link if upload succeeded
        video_id = response.get('id') if response else None
        youtube_url = f'https://youtu.be/{video_id}' if video_id else None
        return response, youtube_url, None
    except Exception as e:
        # Check for quota errors in the error message
        error_msg = str(e)
        if 'quota' in error_msg.lower() or 'quotaexceeded' in error_msg.lower():
            return None, None, 'quota'
        return None, None, error_msg

def load_processed():
    if os.path.exists(PROCESSED_JSON):
        with open(PROCESSED_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_processed(processed):
    with open(PROCESSED_JSON, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

def main():
    parent_dir = get_parent_directory()
    if not parent_dir:
        print("No parent directory selected. Exiting.")
        return
    video_folders = find_video_folders(parent_dir)
    if not video_folders:
        print("No folders with videos found.")
        return
    processed = load_processed()
    youtube = authenticate_youtube()
    total_folders = len(video_folders)
    quota_reached = False
    for idx, folder in enumerate(video_folders, 1):
        # Support both old (flat) and new (parts) processed structures
        skip_folder = False
        if folder in processed:
            entry = processed[folder]
            if isinstance(entry, dict):
                # Old style: flat dict with 'status'
                if 'status' in entry and entry['status'] == 'uploaded':
                    skip_folder = True
                # New style: dict of parts
                elif all(isinstance(v, dict) and v.get('status') == 'uploaded' for v in entry.values()):
                    skip_folder = True
        if skip_folder:
            print(f"[{idx}/{total_folders}] Skipping already processed folder: {folder}")
            continue
        res_groups = find_videos_by_resolution(folder)
        if not res_groups or all(len(vs) == 0 for vs in res_groups.values()):
            processed[folder] = {'status': 'no_videos', 'title': None}
            save_processed(processed)
            continue
        processed[folder] = processed.get(folder, {})
        num_res_groups = len([v for v in res_groups.values() if v])
        part_number = 1
        for res, videos in res_groups.items():
            if not videos:
                continue
            # Determine if we need to use "Part xx" in the title
            use_part = num_res_groups > 1
            part_key_base = f"{res}" if not use_part else f"part_{part_number:02d}_{res}"
            if part_key_base in processed[folder] and processed[folder][part_key_base].get('status') == 'uploaded':
                print(f"[{idx}/{total_folders}] Skipping already processed part: {part_key_base} in {folder}")
                part_number += 1
                continue
            # Title logic
            base_title = get_last_4_folders(folder)
            if res != 'unknown':
                base_title += f" ({res})"
            merged_title = base_title
            if use_part:
                merged_title = f"{base_title} - Part {part_number:02d}"
            merged_path = os.path.join(MERGED_OUTPUT_DIR, merged_title + '.mp4')
            print(f"[{idx}/{total_folders}] Merging {len(videos)} videos in: {folder} (Resolution: {res})")
            try:
                merge_videos_ffmpeg(videos, merged_path)
                print(f"[{idx}/{total_folders}] Merged video saved to: {merged_path}")
                # Check duration and split if needed
                split_proc = subprocess.run([
                    'python', os.path.join(os.path.dirname(__file__), 'split_video_ffmpeg.py'), merged_path, '43200'
                ], capture_output=True, text=True)
                split_files = [merged_path]
                if split_proc.returncode == 0:
                    split_files = [line.strip() for line in split_proc.stdout.splitlines() if line.strip()]
                part_suffix = 'a'
                for split_idx, split_file in enumerate(split_files):
                    this_title = merged_title if len(split_files) == 1 else f"{merged_title}{chr(97+split_idx)}"
                    print(f"[{idx}/{total_folders}] Uploading video to YouTube: {this_title}")
                    response, youtube_url, upload_status = upload_video(youtube, split_file, this_title)
                    split_key = part_key if len(split_files) == 1 else f"{part_key}{chr(97+split_idx)}"
                    if upload_status == 'quota':
                        print(f"[{idx}/{total_folders}] Quota exceeded. Stopping further uploads.")
                        processed[folder][split_key] = {'status': 'quota', 'title': this_title}
                        quota_reached = True
                        save_processed(processed)
                        break
                    elif youtube_url:
                        print(f"[{idx}/{total_folders}] Upload complete: {this_title}")
                        processed[folder][split_key] = {'status': 'uploaded', 'title': this_title, 'youtube_url': youtube_url}
                    else:
                        print(f"[{idx}/{total_folders}] Upload failed: {this_title} ({upload_status})")
                        processed[folder][split_key] = {'status': 'failed', 'title': this_title, 'error': upload_status}
                    save_processed(processed)
                    # Only cleanup and create .txt file if upload succeeded and link is available
                    if processed[folder][split_key].get('status') == 'uploaded' and os.path.exists(split_file):
                        os.remove(split_file)
                        print(f"[{idx}/{total_folders}] Deleted merged video: {split_file}")
                        txt_path = os.path.splitext(split_file)[0] + ".txt"
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(f"YouTube Link: {processed[folder][split_key]['youtube_url']}\n")
                        print(f"[{idx}/{total_folders}] Wrote YouTube link to: {txt_path}")
                    if quota_reached:
                        break
                if quota_reached:
                    break
            except Exception as e:
                print(f"[{idx}/{total_folders}] Failed to process {folder} part {part_number}: {e}")
                processed[folder][part_key] = {'status': 'failed', 'title': merged_title, 'error': str(e)}
                youtube_url = None
            save_processed(processed)
            part_number += 1
        save_processed(processed)
        if quota_reached:
            break
    print("Processing complete. See processed_folders.json for details.")

if __name__ == '__main__':
    main()
