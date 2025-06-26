import os
import sys
import glob
import logging
from datetime import datetime
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import tkinter as tk
from tkinter import filedialog
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import time

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
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 180  # 3 minutes
    for attempt in range(1, MAX_RETRIES + 1):
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
            logging.error(f"Error uploading {video_path} to YouTube (attempt {attempt}): {str(e)}")
            if attempt < MAX_RETRIES:
                logging.info(f"Retrying upload for {video_path} in {RETRY_DELAY_SECONDS // 60} minutes...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(f"Upload failed for {video_path} after {MAX_RETRIES} attempts.")
                return None

def select_folder_dialog(title="Select folder"):
    """Open a folder selection dialog and return the selected path."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        x = (width - 400) // 2
        y = (height - 300) // 2
        root.geometry(f'400x300+{x}+{y}')
        folder_path = filedialog.askdirectory(
            title=title,
            mustexist=True
        )
        root.destroy()
        return folder_path
    except Exception as e:
        logging.error(f"Error in folder selection: {str(e)}")
        return None

def main():
    # Select parent folder
    parent_folder = select_folder_dialog(title="Select parent folder containing videos to upload recursively")
    if not parent_folder or not os.path.exists(parent_folder):
        logging.error("No folder selected or folder does not exist")
        return
    cwd = os.getcwd()
    client_secrets_file = os.path.join(cwd, "client_secret.dlc.json")
    if not os.path.exists(client_secrets_file):
        logging.error(f"Client secrets file not found at: {client_secrets_file}")
        return
    youtube = get_authenticated_service(client_secrets_file)
    logging.info("YouTube authentication complete.")
    VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm')
    for root, _, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith(VIDEO_EXTENSIONS) and not file.endswith('.txt'):
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]
                link_file = os.path.join(root, f"{video_name}.txt")
                if not os.path.exists(link_file):
                    folder_name = os.path.basename(root)
                    MAX_TITLE_LENGTH = 100
                    title = f"{folder_name} - {video_name}"
                    if len(title) > MAX_TITLE_LENGTH:
                        title = title[:MAX_TITLE_LENGTH]
                    logging.info(f"Uploading: {video_path} with title: {title}")
                    video_id = upload_to_youtube(youtube, video_path, title)
                    if video_id:
                        with open(link_file, 'w') as f:
                            f.write(f"https://www.youtube.com/watch?v={video_id}\n")
                        try:
                            os.remove(video_path)
                        except Exception as e:
                            logging.error(f"Failed to delete uploaded video file: {video_path}: {str(e)}")
                        logging.info(f"Upload successful! Video ID: {video_id}")
                    else:
                        logging.error(f"Upload failed for {video_path}")
                else:
                    logging.info(f"Link file already exists for {video_path}. Skipping upload.")
    logging.info("All specified folders and subfolders processed.")

if __name__ == "__main__":
    main() 