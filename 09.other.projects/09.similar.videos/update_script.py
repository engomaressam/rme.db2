import os

# Define the file paths
source_file = r"c:\OneDrive\OneDrive - Rowad Modern Engineering\rme.db\09.other.projects\09.similar.videos\similar_videos_detector_rev05.py"
target_file = r"c:\OneDrive\OneDrive - Rowad Modern Engineering\rme.db\09.other.projects\09.similar.videos\similar_videos_detector_rev06.py"

# Read the entire content of the source file
with open(source_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Define the text to search for (the video information part)
search_text = "            # Video information\n            info_text = f\"Name: {os.path.basename(video.path)}\\n\"\n            info_text += f\"Duration: {video.duration:.2f}s\\n\"\n            info_text += f\"Resolution: {video.resolution}\\n\"\n            info_text += f\"Size: {video.format_size()}\\n\"\n            info_text += f\"Bitrate: {video.bitrate/1000:.1f} Kbps\""

# Define the replacement text with the relative path added
replacement_text = "            # Video information\n            # Get relative path from base directory\n            rel_path = os.path.relpath(os.path.dirname(video.path), self.base_directory)\n            if rel_path == \".\":\n                rel_path = \"<Root Directory>\"\n                \n            info_text = f\"Name: {os.path.basename(video.path)}\\n\"\n            info_text += f\"Path: {rel_path}\\n\"  # Add relative path\n            info_text += f\"Duration: {video.duration:.2f}s\\n\"\n            info_text += f\"Resolution: {video.resolution}\\n\"\n            info_text += f\"Size: {video.format_size()}\\n\"\n            info_text += f\"Bitrate: {video.bitrate/1000:.1f} Kbps\""

# Replace the text
updated_content = content.replace(search_text, replacement_text)

# Write the updated content to the target file
with open(target_file, 'w', encoding='utf-8') as f:
    f.write(updated_content)

print("Successfully created rev06 with relative path display added!")
