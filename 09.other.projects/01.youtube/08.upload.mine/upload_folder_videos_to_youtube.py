import os
import tkinter as tk
from tkinter import filedialog
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import pickle
import mimetypes

VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.mpg', '.mpeg')

def get_youtube_service():
    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        from google.auth.transport.requests import Request
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.emc.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)

def select_directory():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Folder Containing Videos")
    root.destroy()
    return folder

def upload_video(youtube, file_path, title):
    body = {
        "snippet": {"title": title, "description": "Uploaded by script"},
        "status": {"privacyStatus": "private"}
    }
    media = MediaFileUpload(file_path, mimetype=mimetypes.guess_type(file_path)[0], resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()
    video_id = response.get("id")
    return f"https://youtu.be/{video_id}"

def main():
    folder = select_directory()
    if not folder:
        print("No folder selected.")
        return
    youtube = get_youtube_service()
    for filename in os.listdir(folder):
        if not filename.lower().endswith(VIDEO_EXTENSIONS):
            continue
        file_path = os.path.join(folder, filename)
        title = os.path.splitext(filename)[0]
        print(f"Uploading: {filename}")
        try:
            link = upload_video(youtube, file_path, title)
            print(f"Uploaded: {filename} -> {link}")
            os.remove(file_path)
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"YouTube Link: {link}\n")
        except Exception as e:
            print(f"Failed to upload {filename}: {e}")

if __name__ == "__main__":
    main()
