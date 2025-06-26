import os
import glob
import logging
from datetime import datetime
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import tkinter as tk
from tkinter import filedialog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('upload_all.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['PYTHONHTTPSVERIFY'] = '0'

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

def upload_to_youtube(youtube, video_path: str, title: str, description: str = "Uploaded by script"):
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
                logging.info(f"Upload {int(status.progress() * 100)}% complete for {video_path}")
        return response['id']
    except Exception as e:
        logging.error(f"Error uploading {video_path} to YouTube: {str(e)}")
        return None

def select_folder_dialog(title="Select the folder containing videos to upload"):
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_selected

def main():
    cwd = os.getcwd()
    client_secrets_file = os.path.join(cwd, "client_secret.dlc.json")
    if not os.path.exists(client_secrets_file):
        logging.error("Client secrets file not found")
        return
    youtube = get_authenticated_service(client_secrets_file)
    logging.info("YouTube authentication complete.")
    # Ask user for folder containing videos to upload
    input_folder = select_folder_dialog()
    if not input_folder or not os.path.exists(input_folder):
        logging.warning("No folder selected or path does not exist. Using current working directory.")
        input_folder = cwd
    folder_name = os.path.basename(input_folder)
    # Find all video files in the folder (no recursion)
    extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.flv", "*.rmvb")
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    if not video_files:
        logging.info("No video files found in the folder.")
        return
    logging.info(f"Found {len(video_files)} videos to upload.")
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        title = f"{folder_name} - {video_name}"
        logging.info(f"Uploading: {video_path} with title: {title}")
        video_id = upload_to_youtube(youtube, video_path, title)
        if video_id:
            link_file = os.path.join(input_folder, f"{video_name}.txt")
            with open(link_file, 'w') as f:
                f.write(f"https://www.youtube.com/watch?v={video_id}\n")
            try:
                os.remove(video_path)
            except Exception as e:
                logging.error(f"Failed to delete uploaded video file: {video_path}: {str(e)}")
            logging.info(f"Upload successful! Video ID: {video_id}")
        else:
            logging.error(f"Upload failed for {video_path}")

main()
