import os
import pickle
import tkinter as tk
from tkinter import filedialog
import subprocess
import shlex
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
CLIENT_SECRETS_FILE = 'client_secret.emc.json'
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv']
MERGED_OUTPUT_DIR = 'D:/merged'

os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

def get_video_directory():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select Directory Containing Videos To Merge")
    root.destroy()
    return directory

def find_videos(directory):
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, file))
    return sorted(video_files)

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
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploading {file_path}: {int(status.progress() * 100)}% done.")
    print(f"Upload Complete: {file_path}")
    return response

def main():
    video_dir = get_video_directory()
    if not video_dir:
        print("No directory selected. Exiting.")
        return
    videos = find_videos(video_dir)
    if not videos:
        print("No videos found in the selected directory.")
        return
    merged_title = get_last_4_folders(video_dir)
    merged_path = os.path.join(MERGED_OUTPUT_DIR, merged_title + '.mp4')
    print(f"Merging {len(videos)} videos with ffmpeg...")
    merge_videos_ffmpeg(videos, merged_path)
    print(f"Merged video saved to: {merged_path}")
    youtube = authenticate_youtube()
    upload_video(youtube, merged_path, merged_title)

if __name__ == '__main__':
    main()
