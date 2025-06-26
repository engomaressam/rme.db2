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

def process_folder(youtube, input_folder):
    folder_name = os.path.basename(input_folder)
    VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm')
    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                  if os.path.isfile(os.path.join(input_folder, f))
                  and f.lower().endswith(VIDEO_EXTENSIONS)
                  and not f.endswith('.txt')]
    if not video_files:
        logging.info(f"No files found in the folder: {input_folder}")
        return
    logging.info(f"Found {len(video_files)} files to process in {input_folder}.")
    video_files = [f for f in video_files if not os.path.exists(os.path.splitext(f)[0] + '.txt')]
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        MAX_TITLE_LENGTH = 100
        title = f"{folder_name} - {video_name}"
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH]
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

def main():
    # Select parent folder
    parent_folder = select_folder_dialog(title="Select parent folder containing subfolders of videos to upload")
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
    # Find all immediate subfolders
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
                  if os.path.isdir(os.path.join(parent_folder, d))]
    if not subfolders:
        logging.info("No subfolders found in the selected parent folder.")
        return
    for subfolder in subfolders:
        logging.info(f"Processing subfolder: {subfolder}")
        process_folder(youtube, subfolder)
    logging.info("All subfolders processed.")

if __name__ == "__main__":
    main() 