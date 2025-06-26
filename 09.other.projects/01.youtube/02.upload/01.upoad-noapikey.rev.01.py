import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def upload_to_youtube(video_path, title, description, tags=[], privacyStatus='private'):
  """
  Uploads a video to YouTube.

  Args:
      video_path (str): Path to the video file.
      title (str): Title of the video.
      description (str): Description of the video.
      tags (list, optional): List of tags for the video. Defaults to [].
      privacyStatus (str, optional): Privacy status of the video ('private', 'public', 'unlisted'). Defaults to 'private'.
  """

  # Replace with your API key
  api_key = 'YOUR_API_KEY'  
  youtube = build('youtube', 'v3', developerKey=api_key)

  request = youtube.videos().insert(
      part="snippet,status",
      body={
          "snippet": {
              "title": title,
              "description": description,
              "tags": tags
          },
          "status": {
              "privacyStatus": privacyStatus
          }
      },
      media_body=MediaFileUpload(video_path, chunksize=-1, resumable=True)
  )

  response = None
  while response is None:
      status, response = request.next_chunk()
      if status:
          print(f"Uploaded {int(status.progress() * 100)}%")

  print(f"Video uploaded successfully! Video ID: {response['id']}")

def upload_videos_from_folder(folder_path):
  """
  Uploads all videos from a folder to YouTube.

  Args:
      folder_path (str): Path to the folder containing the videos.
  """

  for filename in os.listdir(folder_path):
      if filename.endswith(".mp4"):
          video_path = os.path.join(folder_path, filename)
          title = filename[:-4]  # Remove the .mp4 extension from the filename
          description = f"Uploaded from folder: {folder_path}"
          upload_to_youtube(video_path, title, description)

# Get the folder path
folder_path = input("Enter the folder path containing the videos: ")

# Upload videos from the folder
upload_videos_from_folder(folder_path)