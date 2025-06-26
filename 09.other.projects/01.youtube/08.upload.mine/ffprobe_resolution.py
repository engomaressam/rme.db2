import subprocess
import shlex
import sys

def get_resolution(video_path):
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 \"{video_path}\""
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()

if __name__ == "__main__":
    print(get_resolution(sys.argv[1]))
