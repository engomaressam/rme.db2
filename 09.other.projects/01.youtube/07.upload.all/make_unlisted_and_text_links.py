import os
import re
import logging
import ssl
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('make_unlisted_and_text_links.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

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

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", name)

def get_uploads_playlist_id(youtube):
    channels_response = youtube.channels().list(
        part="contentDetails",
        mine=True
    ).execute()
    for channel in channels_response.get("items", []):
        return channel["contentDetails"]["relatedPlaylists"]["uploads"]
    return None

def get_all_video_ids_from_playlist(youtube, playlist_id):
    video_ids = []
    nextPageToken = None
    while True:
        pl_response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=50,
            pageToken=nextPageToken
        ).execute()
        for item in pl_response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
        nextPageToken = pl_response.get("nextPageToken")
        if not nextPageToken:
            break
    return video_ids

def main():
    cwd = os.getcwd()
    client_secrets_file = os.path.join(cwd, "client_secret.dlc.json")
    if not os.path.exists(client_secrets_file):
        logging.error("Client secrets file not found")
        return
    youtube = get_authenticated_service(client_secrets_file)
    logging.info("YouTube authentication complete.")
    uploads_playlist_id = get_uploads_playlist_id(youtube)
    if not uploads_playlist_id:
        logging.error("Could not get uploads playlist ID.")
        return
    video_ids = get_all_video_ids_from_playlist(youtube, uploads_playlist_id)
    logging.info(f"Found {len(video_ids)} uploaded videos.")
    updated_count = 0
    for video_id in video_ids:
        try:
            vid_response = youtube.videos().list(
                part="id,snippet,status",
                id=video_id
            ).execute()
            if not vid_response["items"]:
                continue
            item = vid_response["items"][0]
            title = item['snippet']['title']
            status = item['status']
            # Only process private (draft) videos
            if status.get('privacyStatus') == 'private':
                youtube.videos().update(
                    part="status",
                    body={
                        "id": video_id,
                        "status": {
                            "privacyStatus": "unlisted",
                            "selfDeclaredMadeForKids": False
                        }
                    }
                ).execute()
                filename = sanitize_filename(title) + ".txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"https://youtu.be/{video_id}\n")
                logging.info(f"Updated to unlisted and saved link for: {title}")
                updated_count += 1
        except Exception as e:
            logging.error(f"Failed to update video {video_id}: {str(e)}")
    logging.info(f"Done. Updated {updated_count} videos.")

if __name__ == "__main__":
    main() 