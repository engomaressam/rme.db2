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



def select_folder_dialog():
    """Open a folder selection dialog and return the selected path."""
    try:
        # Initialize Tkinter
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Set dialog to center of screen
        root.update_idletasks()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        x = (width - 400) // 2
        y = (height - 300) // 2
        root.geometry(f'400x300+{x}+{y}')
        
        # Show folder selection dialog
        folder_path = filedialog.askdirectory(
            title="Select folder containing videos to upload",
            mustexist=True
        )
        
        # Cleanup
        root.destroy()
        return folder_path
    except Exception as e:
        logging.error(f"Error in folder selection: {str(e)}")
        return None

def main():
    # Get folder path from dialog
    input_folder = select_folder_dialog()
    if not input_folder or not os.path.exists(input_folder):
        logging.error("No folder selected or folder does not exist")
        return
        
    cwd = os.getcwd()
    client_secrets_file = os.path.join(cwd, "client_secret.dlc.json")
    if not os.path.exists(client_secrets_file):
        logging.error(f"Client secrets file not found at: {client_secrets_file}")
        return
        
    youtube = get_authenticated_service(client_secrets_file)
    logging.info("YouTube authentication complete.")
    folder_name = os.path.basename(input_folder)
    # Find all files in the folder (no recursion)
    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
                  if os.path.isfile(os.path.join(input_folder, f)) and 
                  not f.endswith('.txt')]  # Skip any existing .txt files
    
    if not video_files:
        logging.info("No files found in the folder.")
        return
        
    logging.info(f"Found {len(video_files)} files to process.")
    
    # Filter out any files that have already been processed
    video_files = [f for f in video_files 
                  if not os.path.exists(os.path.splitext(f)[0] + '.txt')]
                  
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
