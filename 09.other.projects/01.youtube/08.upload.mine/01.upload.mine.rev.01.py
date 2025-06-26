import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# Path to client_secret.json
CLIENT_SECRETS_FILE = 'client_secret.emc.json'  # Updated to new filename

import tkinter as tk
from tkinter import filedialog

def get_video_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title="Select Directory Containing Videos")
    root.destroy()
    return directory

# Allowed video file extensions
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv']

from google.auth.transport.requests import Request

def authenticate_youtube():
    creds = None
    # Use a persistent token file for automation
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('youtube', 'v3', credentials=creds)

def find_videos(directory):
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, file))
    return video_files

def upload_video(youtube, file_path):
    body = {
        'snippet': {
            'title': os.path.basename(file_path),
            'description': 'Uploaded by automated script',
            'tags': [],
            'categoryId': '22'  # People & Blogs
        },
        'status': {
            'privacyStatus': 'private'  # Always upload as private
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
    youtube = authenticate_youtube()
    videos = find_videos(video_dir)
    print(f"Found {len(videos)} video(s) to upload.")
    for video in videos:
        try:
            upload_video(youtube, video)
        except Exception as e:
            print(f"Failed to upload {video}: {e}")

if __name__ == '__main__':
    main()
