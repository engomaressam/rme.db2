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
import logging
import time
import platform
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_detector.log"),
        logging.StreamHandler()
    ]
)

# Directory for temporary files (outside of the repo)
TEMP_DIR = r"D:\d\Download\similar.videos"

# Create temp directory if it doesn't exist
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Create log directory
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging to file
log_file = os.path.join(log_dir, f"video_detector_{time.strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

logging.info("Starting Similar Videos Detector")

class VideoInfo:
    """Store video information and comparison results."""
    def __init__(self, path, duration=None, resolution=None, size=None, bitrate=None):
        self.path = path
        self.duration = duration
        self.resolution = resolution
        self.size = size
        self.bitrate = bitrate
        self.fingerprint = None
        self.thumbnails = []  # List of thumbnail images (PIL format) at different positions (20%, 70%)
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
    def __init__(self, directory, extensions=None, similarity_threshold=0.85, duration_tolerance=1.0, hash_size=8, frame_sampling_density=3, max_duration_diff=3.0, enable_partial_match=False, skip_moviepy_fallback=True):
        self.directory = os.path.abspath(directory)
        if extensions is None:
            self.extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"]
        else:
            self.extensions = extensions
        self.similarity_threshold = similarity_threshold
        self.duration_tolerance = duration_tolerance
        self.hash_size = hash_size  # Controls the precision of image hashing
        self.frame_sampling_density = frame_sampling_density  # Number of frames to sample from each video
        self.max_duration_diff = max_duration_diff  # Maximum duration difference for partial matching (seconds)
        self.enable_partial_match = enable_partial_match  # Enable partial matching between videos with different durations
        self.videos = []
        self.similar_videos = []
        self.video_clusters = []
        self.processed_files = 0
        self.skipped_files = 0
        self.skip_errors = False  # Flag to control error handling behavior
        self.skip_moviepy_fallback = skip_moviepy_fallback  # Flag to skip MoviePy fallback when OpenCV fails
        self.problematic_files = []  # List to track problematic files
        
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
        problematic_files = []
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                # First try with a timeout to avoid hanging on problematic files
                clip = None
                try:
                    # Try to open the video file with a timeout
                    clip = VideoFileClip(video_path, audio=False)  # Skip audio processing to speed up
                    
                    # Extract basic information
                    duration = clip.duration
                    width, height = clip.size
                    resolution = f"{width}x{height}"
                    size = os.path.getsize(video_path)
                    bitrate = size * 8 / duration if duration > 0 else 0
                    
                    # Create video info object
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
                    # Log the specific error
                    error_msg = f"Error processing {video_path}: {str(e)}"
                    logging.error(error_msg)
                    problematic_files.append((video_path, str(e)))
                    self.skipped_files += 1
                
                finally:
                    # Always close the clip if it was opened
                    if clip is not None:
                        try:
                            clip.close()
                        except:
                            pass
                            
            except Exception as e:
                # Catch any other exceptions that might occur
                error_msg = f"Unexpected error with {video_path}: {str(e)}"
                logging.error(error_msg)
                problematic_files.append((video_path, str(e)))
                self.skipped_files += 1
        
        # Report on problematic files
        if problematic_files:
            logging.warning(f"Skipped {len(problematic_files)} files due to errors")
            with open("problematic_videos.log", "w") as f:
                f.write("The following video files were skipped due to errors:\n\n")
                for path, error in problematic_files:
                    f.write(f"{path}\nError: {error}\n\n")
            print(f"\nDetailed list of problematic files saved to 'problematic_videos.log'")
        
        self.videos = valid_videos
        print(f"Successfully extracted info from {self.processed_files} videos")
        if self.skipped_files > 0:
            print(f"Skipped {self.skipped_files} files due to errors")
        return valid_videos
        
    def dhash(self, image, hash_size=None):
        """Compute the difference hash for an image."""
        # Use instance hash_size if none provided
        if hash_size is None:
            hash_size = self.hash_size
            
        # Convert to grayscale and resize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        
        # Calculate differences
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert to hash string
        return sum(2**i for i, v in enumerate(diff.flatten()) if v)
    
    def phash(self, image, hash_size=None):
        """Compute the perceptual hash for an image."""
        # Use instance hash_size if none provided
        if hash_size is None:
            hash_size = self.hash_size
            
        # Convert to grayscale and resize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size, hash_size))
        
        # Calculate DCT (Discrete Cosine Transform)
        dct = cv2.dct(np.float32(resized))
        dct_low_freq = dct[:hash_size, :hash_size]  # Use consistent hash_size
        
        # Calculate median value and generate hash
        med = np.median(dct_low_freq)
        hash_bits = dct_low_freq > med
        
        # Convert to hash
        return sum(2**i for i, v in enumerate(hash_bits.flatten()) if v)
    
    def extract_frame(self, video_path, timestamp):
        """Extract a frame from the video at the specified timestamp."""
        try:
            # First try with OpenCV
            cap = None
            try:
                # Suppress OpenCV error output for h264 reference frame warnings
                original_stderr = None
                null_fd = None
                try:
                    if hasattr(os, 'devnull'):
                        original_stderr = os.dup(2)  # Save stderr file descriptor
                        null_fd = os.open(os.devnull, os.O_WRONLY)
                        os.dup2(null_fd, 2)  # Redirect stderr to /dev/null
                except Exception as e:
                    logging.debug(f"Could not redirect stderr: {str(e)}")
                
                cap = cv2.VideoCapture(video_path)
                
                # Restore stderr if we redirected it
                if original_stderr is not None:
                    os.dup2(original_stderr, 2)
                    os.close(original_stderr)
                if null_fd is not None:
                    os.close(null_fd)
                    
                if not cap.isOpened():
                    raise ValueError(f"Could not open video {video_path}")
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps <= 0 or total_frames <= 0:
                    raise ValueError(f"Invalid video properties for {video_path}")
                
                # Calculate frame number from timestamp
                frame_number = int(timestamp * fps)
                
                # Ensure frame number is within valid range
                frame_number = max(0, min(frame_number, total_frames - 1))
                
                # Set position to the desired frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # Read the frame
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    raise ValueError(f"Could not read frame at {timestamp}s")
                
                return frame
                
            except Exception as e:
                if cap is not None:
                    cap.release()
                
                # Check if we should skip MoviePy fallback
                if self.skip_moviepy_fallback:
                    logging.warning(f"Skipping frame extraction for {video_path}: {str(e)}")
                    return None
                    
                # If we're not skipping MoviePy and not in skip_errors mode, try MoviePy as a fallback
                if not self.skip_errors:
                    logging.warning(f"OpenCV failed to extract frame from {video_path}: {str(e)}. Trying MoviePy...")
                else:
                    # In skip_errors mode, just log and return None
                    logging.warning(f"Skipping frame extraction for {video_path}: {str(e)}")
                    return None
                    
                # Try with MoviePy as fallback
                clip = None
                try:
                    clip = VideoFileClip(video_path, audio=False)
                    frame = clip.get_frame(timestamp)
                    # Convert from RGB to BGR for OpenCV functions
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e2:
                    if clip is not None:
                        try:
                            clip.close()
                        except:
                            pass
                    logging.error(f"Both OpenCV and MoviePy failed to extract frame from {video_path}: {str(e2)}")
                    return None
                finally:
                    if clip is not None:
                        try:
                            clip.close()
                        except:
                            pass
        except Exception as e:
            # Catch any other exceptions that might occur
            error_msg = f"Unexpected error with {video_path}: {str(e)}"
            logging.error(error_msg)
            self.handle_problematic_video(video_path, str(e))
            return None
        
    def handle_problematic_video(self, video_path, error_message):
        """Handle a problematic video by logging it and adding to the problematic_files list."""
        # Add to problematic files list
        self.problematic_files.append((video_path, error_message))
        self.skipped_files += 1
        
        # Log to a separate file for easy reference
        with open(os.path.join(log_dir, "problematic_videos.log"), "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {video_path} - {error_message}\n")

    def calculate_video_fingerprint(self, video_info):
        """Calculate a fingerprint for the video based on sample frames and extract thumbnails."""
        if video_info.fingerprint is not None:
            return video_info.fingerprint
                
        try:
            # Check for valid duration
            if not video_info.duration or video_info.duration <= 0:
                error_msg = f"Invalid duration for {video_info.path}. Skipping fingerprint calculation."
                logging.warning(error_msg)
                if self.skip_errors:
                    self.handle_problematic_video(video_info.path, error_msg)
                    return None
                else:
                    raise ValueError(error_msg)
        
            # Define fixed positions to sample (percentage of video duration)
            # Rev09: Use only 2 positions at 20% and 70% of video duration
            positions = [0.2, 0.7]
            
            # Calculate timestamps for each position
            timestamps = [pos * video_info.duration for pos in positions]
        
            # Extract frames and calculate hashes
            fingerprint = []
            thumbnails = []
            thumbnail_paths = []
            errors = []
            
            for i, timestamp in enumerate(timestamps):
                try:
                    frame = self.extract_frame(video_info.path, timestamp)
                    if frame is None:
                        continue
                    
                    # Calculate perceptual hash and difference hash
                    phash_val = self.phash(frame, self.hash_size)
                    dhash_val = self.dhash(frame, self.hash_size)
                    if phash_val and dhash_val:
                        fingerprint.append((dhash_val, phash_val))
                    
                    # Save thumbnails for both positions
                    # Resize for thumbnail
                    height, width = frame.shape[:2]
                    max_dim = 200
                    if width > height:
                        new_width = max_dim
                        new_height = int(height * (max_dim / width))
                    else:
                        new_height = max_dim
                        new_width = int(width * (max_dim / height))
                    
                    thumbnail = cv2.resize(frame, (new_width, new_height))
                    thumbnails.append(Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)))
                    
                    # Save thumbnail to file
                    thumbnail_filename = os.path.join(TEMP_DIR, f"{uuid.uuid4()}.jpg")
                    cv2.imwrite(thumbnail_filename, thumbnail)
                    thumbnail_paths.append(thumbnail_filename)
                
                except Exception as e:
                    error_msg = f"Error extracting frame at {timestamp}s from {video_info.path}: {str(e)}"
                    logging.warning(error_msg)
                    errors.append(error_msg)
                    continue
        
            if not fingerprint:
                error_msg = f"Could not calculate fingerprint for {video_info.path}"
                if errors:
                    error_msg += f": {'; '.join(errors)}"
                logging.warning(error_msg)
                
                if self.skip_errors:
                    self.handle_problematic_video(video_info.path, error_msg)
                    return None
                else:
                    raise ValueError(error_msg)
            
            # Store thumbnails and paths
            video_info.thumbnails = thumbnails
            video_info.thumbnail_paths = thumbnail_paths
            video_info.fingerprint = fingerprint
            
            return fingerprint
            
        except Exception as e:
            error_msg = f"Failed to calculate fingerprint for {video_info.path}: {str(e)}"
            logging.error(error_msg)
            
            if self.skip_errors:
                self.handle_problematic_video(video_info.path, error_msg)
                return None
            else:
                # When skip_errors is False, we still want to log the problematic video
                # before raising the exception
                self.handle_problematic_video(video_info.path, error_msg)
                raise ValueError(error_msg)
            
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
    
    def compare_partial_match(self, video1, video2):
        """Compare two videos with different durations for partial matching.
        
        This method handles cases where a shorter video might match a segment of a longer video.
        For example, a 7-second clip that matches the first part of a 10-second video.
        """
        # Calculate the duration difference
        duration_diff = abs(video1.duration - video2.duration)
        
        # If the duration difference is too large, skip comparison
        if duration_diff > self.max_duration_diff:
            return 0.0
            
        # Identify shorter and longer video
        shorter_video, longer_video = (video1, video2) if video1.duration < video2.duration else (video2, video1)
        
        # Get fingerprints
        shorter_fingerprint = shorter_video.fingerprint
        longer_fingerprint = longer_video.fingerprint
        
        if not shorter_fingerprint or not longer_fingerprint:
            return 0.0
            
        # For partial matching, we'll try to find the best match between
        # the shorter video and any segment of the longer video
        best_similarity = 0.0
        
        # If we have very few frames, compare them all
        if len(shorter_fingerprint) <= 2 or len(longer_fingerprint) <= 2:
            # Use standard comparison when we don't have enough frames for sliding window
            similarity = self.compare_fingerprints(shorter_fingerprint, longer_fingerprint)
            # Reduce similarity slightly based on duration ratio
            adjusted_similarity = similarity * 0.9 * (shorter_video.duration / longer_video.duration)**0.5
            return adjusted_similarity
        
        # When we have enough frames, use a sliding window approach
        # Assume fingerprints are ordered by frame position
        window_size = len(shorter_fingerprint)
        
        # Try different alignments of the shorter video against the longer one
        for start_idx in range(0, max(1, len(longer_fingerprint) - window_size + 1)):
            # Get the window of frames from the longer video
            window = longer_fingerprint[start_idx:start_idx + window_size]
            
            # Compare this window with the shorter video fingerprint
            similarity = self.compare_fingerprints(shorter_fingerprint, window)
            
            # Keep track of the best matching window
            best_similarity = max(best_similarity, similarity)
        
        # Adjust the similarity based on the duration difference
        # We want to favor cases where the durations are closer to each other
        duration_factor = 1.0 - (duration_diff / self.max_duration_diff) * 0.3
        adjusted_similarity = best_similarity * duration_factor
        
        return adjusted_similarity
    
    def find_similar_videos(self):
        """Find similar videos by duration and visual fingerprint."""
        print("\nFinding similar videos...")
        self.similar_videos = []
        
        # First, extract fingerprints for all videos
        print("Extracting video fingerprints...")
        valid_videos = []
        for video in tqdm(self.videos, desc="Processing videos"):
            try:
                fingerprint = self.calculate_video_fingerprint(video)
                if fingerprint is not None:
                    valid_videos.append(video)
            except ValueError as e:
                # If skip_errors is False, we'll still continue with other videos
                # The error has already been logged by calculate_video_fingerprint
                logging.warning(f"Skipping video due to error: {video.path}")
                continue
        
        # Update videos list to only include valid ones
        self.videos = valid_videos
            
        # Group videos by rounded duration for faster comparison
        duration_groups = defaultdict(list)
        for video in self.videos:
            # Round duration to the nearest duration_tolerance
            rounded_duration = round(video.duration / self.duration_tolerance) * self.duration_tolerance
            duration_groups[rounded_duration].append(video)
        
        print(f"Comparing videos in {len(duration_groups)} duration groups...")
        comparisons = 0
        
        # Compare videos based on configuration
        for duration, group in tqdm(duration_groups.items(), desc="Comparing duration groups"):
            # Get videos from adjacent duration groups (within duration_tolerance)
            adjacent_videos = []
            for d in [duration - self.duration_tolerance, duration + self.duration_tolerance]:
                if d in duration_groups:
                    adjacent_videos.extend(duration_groups[d])
            
            # For partial matching, we'll need videos from a wider range of durations
            partial_match_videos = []
            if self.enable_partial_match:
                # Get videos from groups with duration difference up to max_duration_diff
                for d in duration_groups.keys():
                    if d != duration and abs(d - duration) <= self.max_duration_diff:
                        partial_match_videos.extend(duration_groups[d])
            
            # Compare videos within the same group
            for i in range(len(group)):
                # Compare with other videos in the same group
                for j in range(i+1, len(group)):
                    similarity = self.compare_fingerprints(group[i].fingerprint, group[j].fingerprint)
                    comparisons += 1
                    if similarity >= self.similarity_threshold:
                        self.similar_videos.append((group[i], group[j], similarity))
                
                # Compare with videos in adjacent groups (regular matching)
                for video2 in adjacent_videos:
                    similarity = self.compare_fingerprints(group[i].fingerprint, video2.fingerprint)
                    comparisons += 1
                    if similarity >= self.similarity_threshold:
                        self.similar_videos.append((group[i], video2, similarity))
                
                # Compare with videos from partial matching groups
                if self.enable_partial_match:
                    for video2 in partial_match_videos:
                        # Use the partial matching comparison instead
                        similarity = self.compare_partial_match(group[i], video2)
                        comparisons += 1
                        if similarity >= self.similarity_threshold:
                            self.similar_videos.append((group[i], video2, similarity))
        
        print(f"Performed {comparisons} comparisons")
        print(f"Found {len(self.similar_videos)} similar video pairs\n")
        
        # Group similar videos into clusters
        self.group_similar_videos()
        
        return self.similar_videos
    
    def group_similar_videos(self):
        """Group similar videos into clusters."""
        # Create a graph where nodes are videos and edges are similarities
        graph = defaultdict(set)
        for video1, video2, _ in self.similar_videos:
            graph[video1.path].add(video2.path)
            graph[video2.path].add(video1.path)
        
        # Find connected components (clusters) using DFS
        visited = set()
        clusters = []
        
        for video in self.videos:
            if video.path in visited:
                continue
            
            # Start a new cluster
            cluster = []
            stack = [video]
            visited.add(video.path)
            
            while stack:
                current = stack.pop()
                cluster.append(current)
                
                # Add neighbors to stack
                for neighbor_path in graph[current.path]:
                    if neighbor_path not in visited:
                        neighbor = next((v for v in self.videos if v.path == neighbor_path), None)
                        if neighbor:
                            stack.append(neighbor)
                            visited.add(neighbor_path)
            
            # Sort videos in cluster by quality (higher quality first)
            cluster.sort(key=self.get_video_quality, reverse=True)
            
            # Only add clusters with multiple videos
            if len(cluster) > 1:
                clusters.append(cluster)
        
        self.video_clusters = clusters
        print(f"Grouped similar videos into {len(clusters)} clusters\n")
        return clusters
    
    def get_video_quality(self, video):
        """Calculate video quality based on resolution and bitrate."""
        try:
            width, height = map(int, video.resolution.split('x'))
            return width * height * video.bitrate
        except:
            return 0
    
    def handle_problematic_video(self, video_path, error_message):
        """Handle a problematic video by logging it and adding to the problematic_files list."""
        # Add to the list of problematic files
        self.problematic_files.append((video_path, error_message))
        
        # Log the error
        logging.warning(f"Skipping problematic video: {video_path} - {error_message}")
        
        # Write to the problematic videos log file
        with open("problematic_videos.log", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {video_path} - {error_message}\n")
        
        # Increment the skipped files counter
        self.skipped_files += 1
    
    def save_report(self, output_file=None):
        """Save a report of similar videos to a file in the temp directory."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(TEMP_DIR, f"similar_videos_report_{timestamp}.txt")
        
        # Store the report path as an instance attribute for later reference
        self.report_path = output_file
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Similar Videos Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Directory: {self.directory}\n")
            f.write(f"Total videos processed: {self.processed_files}\n")
            f.write(f"Videos skipped due to errors: {self.skipped_files}\n")
            f.write(f"Similar video pairs found: {len(self.similar_videos)}\n")
            f.write(f"Similarity threshold: {self.similarity_threshold}\n")
            f.write(f"Duration tolerance: {self.duration_tolerance} seconds\n\n")
            
            # Group by clusters
            f.write(f"Found {len(self.video_clusters)} clusters of similar videos:\n\n")
            
            for i, cluster in enumerate(self.video_clusters, 1):
                f.write(f"Cluster {i} ({len(cluster)} videos):\n")
                for video in cluster:
                    f.write(f"  - {video.path}\n")
                    f.write(f"    {video.duration:.2f}s, {video.resolution}, {video.format_size()}, {video.bitrate/1000:.1f}Kbps\n")
                f.write("\n")
            
            # List all similar pairs with similarity score
            f.write("\nDetailed similarity pairs:\n\n")
            for video1, video2, similarity in sorted(self.similar_videos, key=lambda x: x[2], reverse=True):
                f.write(f"Similarity: {similarity:.4f}\n")
                f.write(f"Video 1: {video1.path}\n")
                f.write(f"  {video1.duration:.2f}s, {video1.resolution}, {video1.format_size()}, {video1.bitrate/1000:.1f}Kbps\n")
                f.write(f"Video 2: {video2.path}\n")
                f.write(f"  {video2.duration:.2f}s, {video2.resolution}, {video2.format_size()}, {video2.bitrate/1000:.1f}Kbps\n\n")
        
        print(f"Report saved to {output_file}")
        return output_file


class VideoComparisonGUI:
    """GUI for displaying and comparing similar videos."""
    def __init__(self, root, detector):
        self.root = root
        self.detector = detector
        self.current_cluster_index = 0
        self.current_pair_index = 0
        self.view_mode = "clusters"  # or "pairs"
        self.marked_for_deletion = set()  # Store paths of videos marked for deletion
        
        # Create style for marked videos
        self.style = ttk.Style()
        self.style.configure("Marked.TLabelframe", background="#ffcccc")
        self.style.configure("Marked.TLabelframe.Label", background="#ffcccc", foreground="red", font=(None, 10, "bold"))
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create top control frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # View mode selection
        self.view_mode_label = ttk.Label(self.control_frame, text="View Mode:")
        self.view_mode_label.pack(side=tk.LEFT, padx=5)
        
        self.view_mode_var = tk.StringVar(value="clusters")
        self.clusters_radio = ttk.Radiobutton(self.control_frame, text="Clusters", variable=self.view_mode_var, value="clusters", command=self.change_view_mode)
        self.clusters_radio.pack(side=tk.LEFT, padx=5)
        
        self.pairs_radio = ttk.Radiobutton(self.control_frame, text="All Pairs", variable=self.view_mode_var, value="pairs", command=self.change_view_mode)
        self.pairs_radio.pack(side=tk.LEFT, padx=5)
        
        # Navigation controls
        self.prev_button = ttk.Button(self.control_frame, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(self.control_frame, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(self.control_frame, text="")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Delete selected button
        self.delete_selected_button = ttk.Button(self.control_frame, text="Delete Selected", command=self.delete_selected)
        self.delete_selected_button.pack(side=tk.RIGHT, padx=5)
        
        # Open folder button
        self.open_folder_button = ttk.Button(self.control_frame, text="Open Folder", command=self.open_folder)
        self.open_folder_button.pack(side=tk.RIGHT, padx=5)
        
        # Create scrollable frame for videos
        self.canvas = tk.Canvas(self.main_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add scrollbar
        self.scrollbar = ttk.Scrollbar(self.canvas, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Create frame for video thumbnails and info
        self.videos_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.videos_frame, anchor=tk.NW)
        
        # Configure canvas scrolling
        self.videos_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        # Dictionary to store video frames and widgets
        self.video_widgets = {}
        
        # Initialize the display
        self.update_display()
    
    def change_view_mode(self):
        """Change between cluster view and pair view."""
        self.view_mode = self.view_mode_var.get()
        self.current_cluster_index = 0
        self.current_pair_index = 0
        self.update_display()
    
    def show_previous(self):
        """Show the previous cluster or pair."""
        if self.view_mode == "clusters":
            if self.current_cluster_index > 0:
                self.current_cluster_index -= 1
        else:  # pairs mode
            if self.current_pair_index > 0:
                self.current_pair_index -= 1
        self.update_display()
    
    def show_next(self):
        """Show the next cluster or pair."""
        if self.view_mode == "clusters":
            if self.current_cluster_index < len(self.detector.video_clusters) - 1:
                self.current_cluster_index += 1
        else:  # pairs mode
            if self.current_pair_index < len(self.detector.similar_videos) - 1:
                self.current_pair_index += 1
        self.update_display()
    
    def update_display(self):
        """Update the display based on the current view mode and index."""
        # Clear previous content
        for widget in self.videos_frame.winfo_children():
            widget.destroy()
        self.video_widgets.clear()
        
        if self.view_mode == "clusters":
            self.show_cluster()
        else:  # pairs mode
            self.show_pair()
    
    def show_cluster(self):
        """Show the current cluster."""
        if not self.detector.video_clusters:
            self.status_label.config(text="No clusters found")
            return
        
        # Get current cluster
        cluster = self.detector.video_clusters[self.current_cluster_index]
        
        # Update status
        self.status_label.config(text=f"Cluster {self.current_cluster_index + 1} of {len(self.detector.video_clusters)}")
        
        # Show all videos in the cluster
        for i, video in enumerate(cluster):
            self.add_video_to_display(video, i)
    
    def show_pair(self):
        """Show the current pair."""
        if not self.detector.similar_videos:
            self.status_label.config(text="No similar pairs found")
            return
        
        # Get current pair
        video1, video2, similarity = self.detector.similar_videos[self.current_pair_index]
        
        # Update status
        self.status_label.config(text=f"Pair {self.current_pair_index + 1} of {len(self.detector.similar_videos)} (Similarity: {similarity:.4f})")
        
        # Show videos
        self.add_video_to_display(video1, 0)
        self.add_video_to_display(video2, 1)
    
    def add_video_to_display(self, video, index):
        """Add a video to the display with checkbox for marking."""
        # Create frame for this video
        video_frame = ttk.LabelFrame(self.videos_frame, text=os.path.basename(video.path))
        row = index // 2  # Two videos per row
        col = index % 2
        video_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        # Configure grid weights for the videos_frame
        self.videos_frame.columnconfigure(0, weight=1)
        self.videos_frame.columnconfigure(1, weight=1)
        
        # Create top frame for thumbnails
        top_frame = ttk.Frame(video_frame)
        top_frame.pack(fill=tk.X, expand=True)
        
        # Create frame for both thumbnails side by side
        thumbnails_frame = ttk.Frame(top_frame)
        thumbnails_frame.pack(pady=5)
        
        # Add thumbnails (both 20% and 70% positions)
        thumbnail_labels = []
        for i, thumbnail in enumerate(video.thumbnails[:2]):  # Show up to 2 thumbnails per video
            thumb_frame = ttk.Frame(thumbnails_frame)
            thumb_frame.grid(row=0, column=i, padx=5)
            
            # Add position label
            position_text = "20%" if i == 0 else "70%"
            position_label = ttk.Label(thumb_frame, text=f"Position: {position_text}")
            position_label.pack()
            
            # Add thumbnail
            photo = ImageTk.PhotoImage(thumbnail)
            thumbnail_label = ttk.Label(thumb_frame, image=photo)
            thumbnail_label.image = photo  # Keep a reference to prevent garbage collection
            thumbnail_label.pack(pady=2)
            thumbnail_labels.append(thumbnail_label)
        
        # Info text
        info_frame = ttk.Frame(video_frame)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add checkbox for marking video for deletion
        var = tk.BooleanVar(value=video.path in self.marked_for_deletion)
        check_frame = ttk.Frame(info_frame)
        check_frame.pack(fill=tk.X, pady=2)
        
        check_button = ttk.Checkbutton(
            check_frame, 
            text="Mark for deletion", 
            variable=var,
            command=lambda v=var, p=video.path: self.toggle_mark_for_deletion(p, v.get())
        )
        check_button.pack(side=tk.LEFT)
        
        # Add file size and quality indicator
        quality_frame = ttk.Frame(check_frame)
        quality_frame.pack(side=tk.RIGHT)
        
        size_text = f"Size: {video.format_size()}"
        size_label = ttk.Label(quality_frame, text=size_text)
        size_label.pack(side=tk.RIGHT, padx=5)
        
        # Video info
        info_text = tk.Text(info_frame, height=5, width=40, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, expand=True)
        info_text.insert(tk.END, f"Path: {video.path}\n")
        info_text.insert(tk.END, f"Duration: {video.duration:.2f} seconds\n")
        info_text.insert(tk.END, f"Resolution: {video.resolution}\n")
        info_text.insert(tk.END, f"Bitrate: {video.bitrate/1000:.1f} Kbps\n")
        info_text.configure(state="disabled")  # Make read-only
        
        # Action buttons frame
        action_frame = ttk.Frame(video_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        # Play button
        play_button = ttk.Button(action_frame, text="Play", command=lambda p=video.path: self.play_video_by_path(p))
        play_button.pack(side=tk.LEFT, padx=5)
        
        # Open location button
        open_button = ttk.Button(action_frame, text="Open Location", command=lambda p=video.path: self.open_location_by_path(p))
        open_button.pack(side=tk.LEFT, padx=5)
        
        # Store references
        self.video_widgets[video.path] = {
            "frame": video_frame,
            "thumbnail_labels": thumbnail_labels,
            "info_text": info_text,
            "checkbox_var": var,
            "video": video
        }
    
    def toggle_mark_for_deletion(self, path, is_marked):
        """Mark or unmark a video for deletion."""
        if is_marked:
            self.marked_for_deletion.add(path)
        else:
            self.marked_for_deletion.discard(path)
        
        # Update the status bar with count of marked videos
        self.status_bar.config(text=f"Ready - {len(self.marked_for_deletion)} videos marked for deletion")
        
        # Highlight or unhighlight the frame
        if path in self.video_widgets:
            frame = self.video_widgets[path]["frame"]
            if is_marked:
                frame.configure(style="Marked.TLabelframe")
            else:
                frame.configure(style="TLabelframe")
    
    def play_video_by_path(self, video_path):
        """Play a video using the system's default video player."""
        try:
            if platform.system() == "Windows":
                os.startfile(video_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", video_path])
            else:  # Linux
                subprocess.call(["xdg-open", video_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not play video: {str(e)}")
    
    def open_location_by_path(self, video_path):
        """Open the folder containing the specified video."""
        try:
            folder_path = os.path.dirname(video_path)
            if platform.system() == "Windows":
                subprocess.run(["explorer", folder_path])
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", folder_path])
            else:  # Linux
                subprocess.call(["xdg-open", folder_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open location: {str(e)}")
    
    def delete_video_by_path(self, video_path):
        """Delete the specified video file."""
        try:
            # Confirm deletion
            if not messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete:\n{video_path}?"):
                return
            
            # Delete the file (move to recycle bin/trash)
            if platform.system() == "Windows":
                import winshell
                winshell.delete_file(video_path)
            else:
                from send2trash import send2trash
                send2trash(video_path)
            
            # Remove from marked_for_deletion if it was there
            self.marked_for_deletion.discard(video_path)
            
            # Update the display
            self.update_display()
            
            # Show confirmation
            messagebox.showinfo("Success", "File moved to recycle bin/trash.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete file: {str(e)}")
    
    def delete_selected(self):
        """Delete all videos marked for deletion."""
        if not self.marked_for_deletion:
            messagebox.showinfo("No Selection", "No videos are marked for deletion.")
            return
        
        # Confirm deletion
        count = len(self.marked_for_deletion)
        if not messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete {count} marked videos?"):
            return
        
        # Delete all marked videos
        deleted_count = 0
        failed_paths = []
        
        for path in list(self.marked_for_deletion):  # Create a copy of the set to iterate
            try:
                # Delete the file (move to recycle bin/trash)
                if platform.system() == "Windows":
                    import winshell
                    winshell.delete_file(path)
                else:
                    from send2trash import send2trash
                    send2trash(path)
                
                # Remove from marked_for_deletion
                self.marked_for_deletion.discard(path)
                deleted_count += 1
            except Exception as e:
                failed_paths.append(f"{path}: {str(e)}")
        
        # Update the display
        self.update_display()
        
        # Show results
        if failed_paths:
            error_message = "\n".join(failed_paths)
            messagebox.showerror("Deletion Errors", 
                                f"Successfully deleted {deleted_count} videos.\n\n"
                                f"Failed to delete {len(failed_paths)} videos:\n{error_message}")
        else:
            messagebox.showinfo("Success", f"Successfully moved {deleted_count} videos to recycle bin/trash.")
    
    def open_folder(self):
        """Open the report file."""
        try:
            report_path = self.detector.report_path
            if os.path.exists(report_path):
                # Open the report file with the default text editor
                if platform.system() == "Windows":
                    os.startfile(report_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(["open", report_path])
                else:  # Linux
                    subprocess.call(["xdg-open", report_path])
            else:
                messagebox.showinfo("Report Not Found", "The report file could not be found.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open report: {str(e)}")
    
    def open_folder(self):
        """Open the report file."""
        try:
            report_path = self.detector.report_path
            if os.path.exists(report_path):
                # Open the report file with the default text editor
                if platform.system() == "Windows":
                    os.startfile(report_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.call(["open", report_path])
                else:  # Linux
                    subprocess.call(["xdg-open", report_path])
            else:
                messagebox.showinfo("Report Not Found", "The report file could not be found.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open report: {str(e)}")


def show_parameter_dialog(root=None):
    """Show a dialog to set parameters for the detector."""
    dialog = tk.Toplevel(root) if root else tk.Tk()
    dialog.title("Similar Videos Detector - Parameters")
    dialog.geometry("600x650")
    dialog.resizable(True, True)
    
    # Make dialog modal
    dialog.transient(root) if root else None
    dialog.grab_set()
    
    # Create main frame with padding
    main_frame = ttk.Frame(dialog, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title and description
    title_label = ttk.Label(main_frame, text="Similar Videos Detector", font=("Arial", 16, "bold"))
    title_label.pack(pady=(0, 10))
    
    description_label = ttk.Label(main_frame, text="Configure detection parameters", font=("Arial", 10))
    description_label.pack(pady=(0, 20))
    
    # Create parameter frame
    param_frame = ttk.Frame(main_frame)
    param_frame.pack(fill=tk.BOTH, expand=True)
    
    # Similarity threshold
    ttk.Label(param_frame, text="Similarity Threshold:").grid(row=0, column=0, sticky=tk.W, pady=5)
    similarity_var = tk.DoubleVar(value=0.85)
    similarity_scale = ttk.Scale(param_frame, from_=0.5, to=1.0, length=200, variable=similarity_var)
    similarity_scale.grid(row=0, column=1, sticky=tk.W, pady=5)
    similarity_label = ttk.Label(param_frame, textvariable=tk.StringVar(value="0.85"))
    similarity_label.grid(row=0, column=2, sticky=tk.W, pady=5, padx=5)
    
    def update_similarity_label(*args):
        similarity_label.config(text=f"{similarity_var.get():.2f}")
    
    similarity_var.trace_add("write", update_similarity_label)
    
    # Duration tolerance
    ttk.Label(param_frame, text="Duration Tolerance (seconds):").grid(row=1, column=0, sticky=tk.W, pady=5)
    tolerance_var = tk.DoubleVar(value=1.0)
    tolerance_spinbox = ttk.Spinbox(param_frame, from_=0.1, to=10.0, increment=0.1, textvariable=tolerance_var, width=10)
    tolerance_spinbox.grid(row=1, column=1, sticky=tk.W, pady=5)
    ttk.Label(param_frame, text="How close video durations must be to compare").grid(row=1, column=2, sticky=tk.W, pady=5, padx=5)
    
    # Hash size
    ttk.Label(param_frame, text="Hash Size:").grid(row=2, column=0, sticky=tk.W, pady=5)
    hash_var = tk.IntVar(value=8)
    hash_spinbox = ttk.Spinbox(param_frame, from_=4, to=16, increment=2, textvariable=hash_var, width=10)
    hash_spinbox.grid(row=2, column=1, sticky=tk.W, pady=5)
    ttk.Label(param_frame, text="Larger values = more precise but slower").grid(row=2, column=2, sticky=tk.W, pady=5, padx=5)
    
    # Frame sampling density
    ttk.Label(param_frame, text="Frame Sampling Density:").grid(row=3, column=0, sticky=tk.W, pady=5)
    density_var = tk.IntVar(value=2)  # Changed to 2 for rev09
    density_spinbox = ttk.Spinbox(param_frame, from_=1, to=10, increment=1, textvariable=density_var, width=10, state="disabled")
    density_spinbox.grid(row=3, column=1, sticky=tk.W, pady=5)
    ttk.Label(param_frame, text="Fixed at 2 frames in rev09 (20% and 70% positions)").grid(row=3, column=2, sticky=tk.W, pady=5, padx=5)
    
    # Max duration difference for partial matching
    ttk.Label(param_frame, text="Max Duration Difference:").grid(row=4, column=0, sticky=tk.W, pady=5)
    max_diff_var = tk.DoubleVar(value=3.0)
    max_diff_spinbox = ttk.Spinbox(param_frame, from_=1.0, to=30.0, increment=0.5, textvariable=max_diff_var, width=10)
    max_diff_spinbox.grid(row=4, column=1, sticky=tk.W, pady=5)
    ttk.Label(param_frame, text="Maximum duration difference for partial matching").grid(row=4, column=2, sticky=tk.W, pady=5, padx=5)
    
    # Enable partial content matching
    ttk.Label(param_frame, text="Enable Partial Matching:").grid(row=5, column=0, sticky=tk.W, pady=5)
    partial_var = tk.BooleanVar(value=False)
    partial_check = ttk.Checkbutton(param_frame, variable=partial_var)
    partial_check.grid(row=5, column=1, sticky=tk.W, pady=5)
    ttk.Label(param_frame, text="Match videos with different durations").grid(row=5, column=2, sticky=tk.W, pady=5, padx=5)
    
    # Skip MoviePy fallback
    ttk.Label(param_frame, text="Skip MoviePy Fallback:").grid(row=6, column=0, sticky=tk.W, pady=5)
    skip_moviepy_var = tk.BooleanVar(value=True)
    skip_moviepy_check = ttk.Checkbutton(param_frame, variable=skip_moviepy_var)
    skip_moviepy_check.grid(row=6, column=1, sticky=tk.W, pady=5)
    ttk.Label(param_frame, text="Skip MoviePy fallback after OpenCV failure (faster)").grid(row=6, column=2, sticky=tk.W, pady=5, padx=5)
    
    # Advanced options section
    ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
    
    advanced_label = ttk.Label(main_frame, text="Advanced Options", font=("Arial", 10, "bold"))
    advanced_label.pack(anchor=tk.W, pady=(0, 10))
    
    advanced_frame = ttk.Frame(main_frame)
    advanced_frame.pack(fill=tk.BOTH, expand=True)
    
    # Skip errors
    ttk.Label(advanced_frame, text="Skip Errors:").grid(row=0, column=0, sticky=tk.W, pady=5)
    skip_errors_var = tk.BooleanVar(value=True)
    skip_errors_check = ttk.Checkbutton(advanced_frame, variable=skip_errors_var)
    skip_errors_check.grid(row=0, column=1, sticky=tk.W, pady=5)
    ttk.Label(advanced_frame, text="Skip problematic files instead of stopping").grid(row=0, column=2, sticky=tk.W, pady=5, padx=5)
    
    # Information section
    ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
    
    info_frame = ttk.Frame(main_frame)
    info_frame.pack(fill=tk.BOTH, expand=True)
    
    info_text = "Rev09: This version extracts only 2 thumbnails per video at 20% and 70% of the video duration for faster processing."
    info_label = ttk.Label(info_frame, text=info_text, wraplength=550, justify=tk.LEFT)
    info_label.pack(fill=tk.X, pady=5)
    
    # Button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=20)
    
    # Cancel button
    cancel_button = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
    cancel_button.pack(side=tk.RIGHT, padx=5)
    
    # OK button
    params = {}
    
    def on_ok():
        # Store parameters
        params["similarity_threshold"] = similarity_var.get()
        params["duration_tolerance"] = tolerance_var.get()
        params["hash_size"] = hash_var.get()
        params["frame_sampling_density"] = 2  # Fixed at 2 for rev09
        params["max_duration_diff"] = max_diff_var.get()
        params["enable_partial_match"] = partial_var.get()
        params["skip_moviepy_fallback"] = skip_moviepy_var.get()
        params["skip_errors"] = skip_errors_var.get()
        dialog.destroy()
    
    ok_button = ttk.Button(button_frame, text="OK", command=on_ok)
    ok_button.pack(side=tk.RIGHT, padx=5)
    
    # Center the dialog on the parent window
    if root:
        dialog.update_idletasks()
        x = root.winfo_x() + (root.winfo_width() - dialog.winfo_width()) // 2
        y = root.winfo_y() + (root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    # Wait for the dialog to close
    dialog.wait_window()
    
    return params


def main(root=None):
    """Main function to run the detector."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect similar videos in a directory.")
    parser.add_argument("--dir", help="Directory to scan for videos")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Duration tolerance in seconds")
    parser.add_argument("--skip-errors", action="store_true", help="Skip problematic files instead of stopping")
    args = parser.parse_args()
    
    # Get directory from command line or ask user
    directory = args.dir
    if not directory:
        if root:
            root.withdraw()  # Hide the main window temporarily
        directory = filedialog.askdirectory(title="Select Directory with Videos")
        if root:
            root.deiconify()  # Show the main window again
        
        if not directory:  # User cancelled
            print("No directory selected. Exiting.")
            return
    
    # Show parameter dialog
    params = show_parameter_dialog(root)
    if not params:  # User cancelled
        return
    
    logging.info(f"Parameters: similarity_threshold={params['similarity_threshold']}, "
                f"duration_tolerance={params['duration_tolerance']}, "
                f"hash_size={params['hash_size']}, "
                f"frame_sampling_density={params['frame_sampling_density']}, "
                f"enable_partial_match={params['enable_partial_match']}, "
                f"max_duration_diff={params['max_duration_diff']}, "
                f"skip_moviepy_fallback={params['skip_moviepy_fallback']}")
    
    # Apply parameters
    detector = SimilarVideosDetector(
        directory=directory,
        similarity_threshold=params.get("similarity_threshold", args.threshold),
        duration_tolerance=params.get("duration_tolerance", args.tolerance),
        hash_size=params.get("hash_size", 8),
        frame_sampling_density=params.get("frame_sampling_density", 2),  # Fixed at 2 for rev09
        max_duration_diff=params.get("max_duration_diff", 3.0),
        enable_partial_match=params.get("enable_partial_match", False),
        skip_moviepy_fallback=params.get("skip_moviepy_fallback", True)
    )
    
    # Set skip_errors flag
    detector.skip_errors = params.get("skip_errors", args.skip_errors)
    if detector.skip_errors:
        logging.info("Skip errors mode enabled - problematic files will be skipped")
    else:
        logging.info("Skip errors mode disabled - problematic files will cause processing to stop")
    
    video_files = detector.scan_directory()
    if not video_files:
        print("No video files found. Exiting.")
        return
    
    detector.extract_video_info(video_files)
    if not detector.videos:
        print("No valid videos found. Exiting.")
        return
    
    detector.find_similar_videos()
    if not detector.similar_videos:
        print("No similar videos found. Exiting.")
        return
    
    # Save report
    detector.save_report()
    
    # Launch GUI
    if root is None:
        root = tk.Tk()
        root.title("Similar Videos Detector")
        root.geometry("1200x800")
        root.protocol("WM_DELETE_WINDOW", root.quit)
    
    gui = VideoComparisonGUI(root, detector)
    
    if root.winfo_exists():
        root.mainloop()


if __name__ == "__main__":
    # Configure logging
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join("logs", f"similar_videos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler()
        ]
    )
    
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logging.exception("An error occurred in the main function")
        messagebox.showerror("Error", f"An error occurred: {str(e)}\n\nSee logs for details.")
    finally:
        # Clean up temporary files
        try:
            for file in os.listdir(TEMP_DIR):
                if file.endswith(".jpg"):
                    os.remove(os.path.join(TEMP_DIR, file))
        except Exception as e:
            logging.error(f"Error cleaning up temporary files: {str(e)}")
        
        logging.info("Application finished")