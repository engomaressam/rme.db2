import subprocess
import sys
import os

def split_video(input_path, max_seconds=43200):
    # Get duration using ffprobe
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print('ffprobe error:', result.stderr)
        return []
    duration = float(result.stdout.strip())
    if duration <= max_seconds:
        return [input_path]
    # Split the video
    parts = []
    base, ext = os.path.splitext(input_path)
    idx = 0
    start = 0
    while start < duration:
        out_path = f"{base}_split{chr(97+idx)}{ext}"
        cmd = [
            'ffmpeg', '-y', '-i', input_path, '-ss', str(int(start)), '-t', str(int(min(max_seconds, duration-start))), '-c', 'copy', out_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print('ffmpeg split error:', result.stderr)
            break
        parts.append(out_path)
        idx += 1
        start += max_seconds
    return parts

if __name__ == "__main__":
    input_path = sys.argv[1]
    max_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 43200
    for part in split_video(input_path, max_seconds):
        print(part)
