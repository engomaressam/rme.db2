#!/usr/bin/env python3
import os
import glob
import json
import time
import shutil
import logging
import re
from datetime import datetime
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import tkinter as tk
from tkinter import ttk, messagebox
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
import numpy as np
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler('video_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

class ProgressWindow:
    def __init__(self, title="Progress", parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title(title)
        self.root.geometry("600x200")
        
        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'+{x}+{y}')
        
        # Make window stay on top
        self.root.attributes('-topmost', True)
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            main_frame, 
            variable=self.progress_var,
            maximum=100,
            length=500
        )
        self.progress_bar.pack(pady=5)
        
        # Details frame
        details_frame = ttk.LabelFrame(main_frame, text="Details", padding="5")
        details_frame.pack(fill=tk.X, pady=5)
        
        # Time info
        time_frame = ttk.Frame(details_frame)
        time_frame.pack(fill=tk.X, pady=2)
        
        self.elapsed_var = tk.StringVar(value="Elapsed: 0:00:00")
        self.remaining_var = tk.StringVar(value="Remaining: --:--:--")
        
        ttk.Label(time_frame, textvariable=self.elapsed_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(time_frame, textvariable=self.remaining_var).pack(side=tk.RIGHT, padx=5)
        
        # Stats frame
        stats_frame = ttk.Frame(details_frame)
        stats_frame.pack(fill=tk.X, pady=2)
        
        self.speed_var = tk.StringVar(value="Speed: --")
        self.size_var = tk.StringVar(value="Size: --")
        
        ttk.Label(stats_frame, textvariable=self.speed_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(stats_frame, textvariable=self.size_var).pack(side=tk.RIGHT, padx=5)
        
        self.start_time = time.time()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.allow_close = False

    def on_closing(self):
        if self.allow_close:
            self.root.destroy()

    def update_progress(self, current: float, total: float, 
                       speed: str = "", size: str = "", status: str = None):
        if status:
            self.status_var.set(status)
        
        progress = (current / total * 100) if total > 0 else 0
        self.progress_var.set(progress)
        
        elapsed = time.time() - self.start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        self.elapsed_var.set(f"Elapsed: {elapsed_str}")
        
        if progress > 0:
            total_estimated = elapsed / (progress / 100)
            remaining = total_estimated - elapsed
            remaining_str = time.strftime('%H:%M:%S', time.gmtime(remaining))
            self.remaining_var.set(f"Remaining: {remaining_str}")
        
        if speed:
            self.speed_var.set(f"Speed: {speed}")
        if size:
            self.size_var.set(f"Size: {size}")
        
        self.root.update()

    def close(self):
        self.allow_close = True
        self.root.destroy()

def get_video_info(video_path: str) -> Optional[Dict]:
    """Get video information using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
        }
        cap.release()
        return info
    except Exception as e:
        logging.error(f"Error getting video info: {str(e)}")
        return None

def videos_are_compatible(video_files: List[str]) -> bool:
    """Check if videos can be concatenated directly."""
    if not video_files:
        return False
        
    try:
        base_info = get_video_info(video_files[0])
        if not base_info:
            return False
            
        for video_file in video_files[1:]:
            info = get_video_info(video_file)
            if not info:
                return False
                
            # Check key parameters
            if (info['width'] != base_info['width'] or
                info['height'] != base_info['height'] or
                abs(info['fps'] - base_info['fps']) > 0.1 or  # Allow small FPS difference
                info['fourcc'] != base_info['fourcc']):
                return False
                
        return True
    except Exception as e:
        logging.error(f"Error checking video compatibility: {str(e)}")
        return False

def direct_concatenate(video_files: List[str], output_path: str, 
                      progress_window: Optional[ProgressWindow] = None) -> bool:
    """Concatenate videos directly by copying bytes."""
    try:
        total_size = sum(os.path.getsize(f) for f in video_files)
        current_size = 0
        
        with open(output_path, 'wb') as outfile:
            for video_file in video_files:
                # Get file size for progress calculation
                file_size = os.path.getsize(video_file)
                
                with open(video_file, 'rb') as infile:
                    while True:
                        # Read in chunks of 1MB
                        chunk = infile.read(1024 * 1024)
                        if not chunk:
                            break
                            
                        outfile.write(chunk)
                        current_size += len(chunk)
                        
                        if progress_window:
                            progress_window.update_progress(
                                current_size,
                                total_size,
                                size=f"{current_size // (1024*1024)}MB / {total_size // (1024*1024)}MB",
                                status="Concatenating videos..."
                            )
        
        return True
    except Exception as e:
        logging.error(f"Error in direct concatenation: {str(e)}")
        return False

def merge_with_moviepy(video_files: List[str], output_path: str,
                      progress_window: Optional[ProgressWindow] = None) -> bool:
    """Merge videos using MoviePy."""
    clips = []
    total_duration = 0
    current_duration = 0
    
    try:
        # First pass: calculate total duration and verify files
        print("Calculating total duration...")
        for video_file in video_files:
            clip = VideoFileClip(video_file)
            total_duration += clip.duration
            clip.close()
        
        if total_duration == 0:
            logging.error("Total duration is 0")
            return False
            
        print(f"Total duration: {total_duration:.2f} seconds")
        
        # Second pass: actual merging
        clips = []
        for video_file in video_files:
            if progress_window:
                progress_window.update_progress(
                    current_duration,
                    total_duration,
                    status=f"Loading {os.path.basename(video_file)}..."
                )
            
            clip = VideoFileClip(video_file)
            clips.append(clip)
            current_duration += clip.duration
        
        if progress_window:
            progress_window.update_progress(
                current_duration,
                total_duration,
                status="Concatenating clips..."
            )
        
        # Concatenate and write
        final_clip = concatenate_videoclips(clips, method="compose")
        
        if progress_window:
            progress_window.update_progress(
                current_duration,
                total_duration,
                status="Writing final video..."
            )
        
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            logger=None  # Disable moviepy's console output
        )
        
        return True
        
    except Exception as e:
        logging.error(f"Error in MoviePy merge: {str(e)}")
        return False
    finally:
        # Clean up clips
        for clip in clips:
            try:
                clip.close()
            except:
                pass

def merge_videos_alternative(video_files: List[Tuple[str, str]], output_path: str,
                           progress_window: Optional[ProgressWindow] = None) -> bool:
    """Alternative video merging implementation."""
    try:
        # Extract full paths
        video_paths = [full_path for full_path, _ in video_files]
        
        # First try direct concatenation if videos are compatible
        if videos_are_compatible(video_paths):
            logging.info("Videos are compatible for direct concatenation")
            if direct_concatenate(video_paths, output_path, progress_window):
                logging.info("Direct concatenation successful")
                return True
            logging.warning("Direct concatenation failed, falling back to MoviePy")
        
        # Fall back to MoviePy
        logging.info("Using MoviePy for video merging")
        return merge_with_moviepy(video_paths, output_path, progress_window)
        
    except Exception as e:
        logging.error(f"Error in alternative merge: {str(e)}")
        return False

# ... [Rest of the code remains the same as rev.08, just replace merge_videos with merge_videos_alternative] ... 