import os
import sys
import cv2
import numpy as np
import hashlib
import argparse
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import shutil
import uuid
from moviepy.video.io.VideoFileClip import VideoFileClip

# Directory for temporary files (outside of the repo)
TEMP_DIR = r"D:\d\Download\similar.videos"
os.makedirs(TEMP_DIR, exist_ok=True)

class VideoInfo:
    """Store video information and comparison results."""
    def __init__(self, path, duration=None, resolution=None, size=None, bitrate=None):
        self.path = path
        self.duration = duration
        self.resolution = resolution
        self.size = size
        self.bitrate = bitrate
        self.fingerprint = None
        self.thumbnails = []  # List of thumbnail images (PIL format) at different positions (10%, 50%, 90%)
        self.thumbnail_paths = []  # Paths to saved thumbnail files

    def __str__(self):
        return f"{os.path.basename(self.path)} - {self.duration:.2f}s, {self.resolution}, {self.format_size()}, {self.bitrate/1000:.1f}Kbps"
    
    def format_size(self):
        """Format file size in human-readable format."""
        size_bytes = self.size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024 or unit == 'GB':
                return f"{size_bytes:.2f}{unit}"
            size_bytes /= 1024


class SimilarVideosDetector:
    """Detect similar videos in a directory and its subdirectories."""
    def __init__(self, directory, extensions=None, similarity_threshold=0.85, duration_tolerance=1.0):
        self.directory = os.path.abspath(directory)
        if extensions is None:
            self.extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"]
        else:
            self.extensions = extensions
        self.similarity_threshold = similarity_threshold
        self.duration_tolerance = duration_tolerance
        self.videos = []
        self.similar_videos = []
        self.video_clusters = []
        self.processed_files = 0
        self.skipped_files = 0
        
    def scan_directory(self):
        """Scan directory and subdirectories for video files."""
        print(f"Scanning directory: {self.directory}")
        video_files = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.extensions):
                    video_files.append(os.path.join(root, file))
        print(f"Found {len(video_files)} video files")
        return video_files
    
    def extract_video_info(self, video_files):
        """Extract information from video files."""
        print("Extracting video information...")
        valid_videos = []
        self.processed_files = 0
        self.skipped_files = 0
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                with VideoFileClip(video_path) as clip:
                    duration = clip.duration
                    width, height = clip.size
                    resolution = f"{width}x{height}"
                    size = os.path.getsize(video_path)
                    bitrate = size * 8 / duration if duration > 0 else 0
                    
                    video_info = VideoInfo(
                        path=video_path,
                        duration=duration,
                        resolution=resolution,
                        size=size,
                        bitrate=bitrate
                    )
                    valid_videos.append(video_info)
                    self.processed_files += 1
            except Exception as e:
                print(f"\nError processing {video_path}: {str(e)}")
                self.skipped_files += 1
        
        self.videos = valid_videos
        print(f"Successfully extracted info from {self.processed_files} videos")
        if self.skipped_files > 0:
            print(f"Skipped {self.skipped_files} files due to errors")
        return valid_videos
        
    def dhash(self, image, hash_size=8):
        """Compute the difference hash for an image."""
        # Convert to grayscale and resize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        
        # Calculate differences
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert to hash string
        return sum(2**i for i, v in enumerate(diff.flatten()) if v)
    
    def phash(self, image, hash_size=8):
        """Compute the perceptual hash for an image."""
        # Convert to grayscale and resize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size, hash_size))
        
        # Calculate DCT (Discrete Cosine Transform)
        dct = cv2.dct(np.float32(resized))
        dct_low_freq = dct[:8, :8]
        
        # Calculate median value and generate hash
        med = np.median(dct_low_freq)
        hash_bits = dct_low_freq > med
        
        # Convert to hash
        return sum(2**i for i, v in enumerate(hash_bits.flatten()) if v)
    
    def extract_frame(self, video_path, timestamp):
        """Extract a frame from the video at the specified timestamp."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the frame number based on timestamp
        frame_number = int(timestamp * fps)
        
        # Ensure frame number is within bounds
        frame_number = min(max(0, frame_number), total_frames - 1)
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        
        # Release the capture
        cap.release()
        
        if ret:
            return frame
        return None
    
    def calculate_video_fingerprint(self, video_info):
        """Calculate a fingerprint for the video based on sample frames and extract thumbnails."""
        if video_info.fingerprint is not None:
            return video_info.fingerprint
            
        try:
            cap = cv2.VideoCapture(video_info.path)
            if not cap.isOpened():
                raise Exception("Could not open video")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise Exception("Invalid frame count")
            
            # Extract frames at 10%, 50%, and 90% of the video duration
            positions = [0.1, 0.5, 0.9]
            frame_hashes = []
            video_info.thumbnails = []
            video_info.thumbnail_paths = []
            
            for pos in positions:
                frame_idx = int(total_frames * pos)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Generate hashes for similarity comparison
                dhash_val = self.dhash(frame)
                phash_val = self.phash(frame)
                frame_hashes.append((dhash_val, phash_val))
                
                # Save thumbnail to temporary directory
                unique_id = str(uuid.uuid4())[:8]
                file_name = f"{os.path.splitext(os.path.basename(video_info.path))[0]}_{pos}_{unique_id}.jpg"
                thumbnail_path = os.path.join(TEMP_DIR, file_name)
                
                # Resize for consistent thumbnails
                thumbnail = cv2.resize(frame, (320, 180))
                cv2.imwrite(thumbnail_path, thumbnail)
                
                # Convert from BGR to RGB for PIL
                rgb_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_thumbnail)
                
                # Store both the PIL image and the path
                video_info.thumbnails.append(pil_image)
                video_info.thumbnail_paths.append(thumbnail_path)
            
            cap.release()
            
            # Store the fingerprint
            video_info.fingerprint = frame_hashes
            return frame_hashes
            
        except Exception as e:
            print(f"Error calculating fingerprint for {video_info.path}: {str(e)}")
            return None
    
    def hamming_distance(self, hash1, hash2):
        """Calculate the Hamming distance between two hash values."""
        # XOR the hashes and count the number of set bits (1s)
        return bin(hash1 ^ hash2).count('1')
        
    def compare_fingerprints(self, fingerprint1, fingerprint2):
        """Compare two video fingerprints and return similarity score (0-1)."""
        if not fingerprint1 or not fingerprint2:
            return 0.0
        
        # Find the maximum similarity between any pair of frames
        max_similarity = 0.0
        
        for f1 in fingerprint1:
            for f2 in fingerprint2:
                dhash_distance = self.hamming_distance(f1[0], f2[0])
                phash_distance = self.hamming_distance(f1[1], f2[1])
                
                # Normalize distances (64-bit hash has max distance of 64)
                dhash_similarity = 1 - (dhash_distance / 64)
                phash_similarity = 1 - (phash_distance / 64)
                
                # Average of both hash similarities
                avg_similarity = (dhash_similarity + phash_similarity) / 2
                max_similarity = max(max_similarity, avg_similarity)
        
        return max_similarity
