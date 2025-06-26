import os
os.environ['PYTHONHTTPSVERIFY'] = '0'
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd
import isodate  # Convert ISO 8601 duration to HH:MM:SS
from datetime import datetime  # Convert published date to Excel-friendly format
import httplib2

# Define OAuth 2.0 scope (Read-only access to YouTube)
SCOPES = ["https://www.googleapis.com/auth/youtube.readonly"]

# Authenticate using OAuth 2.0
flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
creds = flow.run_local_server(port=0)

# Build YouTube API client
# http = httplib2.Http(disable_ssl_certificate_validation=True)
youtube = build("youtube", "v3", credentials=creds)

# Function to get Uploads Playlist ID for the channel
def get_uploads_playlist_id():
    request = youtube.channels().list(
        part="contentDetails",
        id="UCVjCW3tx7MT24MikTJXOZIw"  # ✅ Your Fixed Channel ID
    )
    response = request.execute()

    if "items" not in response or not response["items"]:
        raise Exception("❌ Error: No channel data found! Double-check the Channel ID.")

    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

# Function to fetch all videos with additional details
def get_videos():
    playlist_id = get_uploads_playlist_id()
    videos = []
    next_page_token = None

    while True:
        request = youtube.playlistItems().list(
            part="snippet",
            playlistId=playlist_id,  # Fetch videos from Uploads playlist
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        video_ids = [item["snippet"]["resourceId"]["videoId"] for item in response.get("items", [])]

        # Fetch video details (duration, description, publish date, license, privacy, region restriction)
        details_request = youtube.videos().list(
            part="contentDetails,snippet,status",
            id=",".join(video_ids)  # Fetch details for multiple videos in one call
        )
        details_response = details_request.execute()

        for item in details_response.get("items", []):
            title = item["snippet"]["title"]
            video_id = item["id"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            duration = isodate.parse_duration(item["contentDetails"]["duration"])  # Convert ISO duration to HH:MM:SS
            description = item["snippet"]["description"].replace("\n", " ")  # Remove line breaks for cleaner Excel output
            license_type = item["status"].get("license", "Unknown")  # License type (Creative Commons / Standard YouTube License)
            privacy_status = item["status"].get("privacyStatus", "Unknown")  # Public, Private, or Unlisted

            # ✅ Convert Published Date to Excel-Friendly Format
            raw_published_at = item["snippet"]["publishedAt"]  # Example: 2025-02-11T13:40:36Z
            published_at = datetime.strptime(raw_published_at, "%Y-%m-%dT%H:%M:%SZ").strftime("%d-%b-%y")  # Output: 12-Feb-25

            # Check for region restrictions
            region_restriction = item["contentDetails"].get("regionRestriction", {}).get("blocked", [])
            if region_restriction:
                blocked_countries = ", ".join(region_restriction)
            else:
                blocked_countries = "None"

            videos.append([title, video_url, str(duration), published_at, description, license_type, privacy_status, blocked_countries])

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return videos

# Fetch videos
video_list = get_videos()

# Save to Excel
df = pd.DataFrame(video_list, columns=["Title", "URL", "Duration", "Published Date", "Description", "License", "Privacy Status", "Blocked Countries"])
df.to_excel("YouTube_Video_List.xlsx", index=False)

print("✅ YouTube video list with formatted dates saved as 'YouTube_Video_List.xlsx'")
