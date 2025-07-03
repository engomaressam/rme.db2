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
    def __init__(self, directory, extensions=None, similarity_threshold=0.85, duration_tolerance=1.0, hash_size=8, frame_sampling_density=3, max_duration_diff=10.0, enable_partial_match=True):
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
            
            # Get original aspect ratio
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            aspect_ratio = orig_width / orig_height if orig_height > 0 else 16/9  # Default if can't determine
            
            # Extract frames based on frame_sampling_density
            # Create evenly distributed positions based on the sampling density
            positions = [i/(self.frame_sampling_density-1) for i in range(self.frame_sampling_density)] if self.frame_sampling_density > 1 else [0.5]
            # Adjust first and last to be slightly inset from the edges
            if len(positions) > 1:
                positions[0] = 0.1  # First position at 10%
                positions[-1] = 0.9  # Last position at 90%
                
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
                
                # Save thumbnail to temporary directory with preserved aspect ratio
                unique_id = str(uuid.uuid4())[:8]
                file_name = f"{os.path.splitext(os.path.basename(video_info.path))[0]}_{pos}_{unique_id}.jpg"
                thumbnail_path = os.path.join(TEMP_DIR, file_name)
                
                # Determine thumbnail dimensions while preserving aspect ratio
                thumb_width = 320  # Target width
                thumb_height = int(thumb_width / aspect_ratio)  # Calculate height to maintain aspect ratio
                
                # Resize while maintaining aspect ratio
                thumbnail = cv2.resize(frame, (thumb_width, thumb_height))
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
        for video in tqdm(self.videos, desc="Processing videos"):
            self.calculate_video_fingerprint(video)
        
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
    
    def save_report(self, output_file=None):
        """Save a report of similar videos to a file in the temp directory."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(TEMP_DIR, f"similar_videos_report_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Similar Videos Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {self.directory}\n")
            f.write(f"Found {len(self.similar_videos)} similar video pairs in {len(self.video_clusters)} groups\n\n")
            
            for i, cluster in enumerate(self.video_clusters, 1):
                f.write(f"Group {i} - {len(cluster)} videos:\n")
                for j, video in enumerate(cluster):
                    quality_diff = 0.0
                    if j > 0 and j < len(cluster):
                        # Calculate quality difference compared to best quality video
                        best_quality = self.get_video_quality(cluster[0])
                        current_quality = self.get_video_quality(video)
                        if best_quality > 0:
                            quality_diff = (best_quality - current_quality) / best_quality * 100
                    
                    keep_delete = "[KEEP]" if j == 0 else f"[POTENTIAL DELETE] (-{quality_diff:.1f}% quality)"
                    f.write(f"  {keep_delete} {video}\n")
                f.write("\n")
        
        print(f"Report saved to {output_file}")
        return output_file


class VideoComparisonGUI:
    """GUI for comparing similar videos with multiple thumbnails per video."""
    def __init__(self, master, detector):
        self.master = master
        self.detector = detector
        self.clusters = detector.video_clusters
        self.current_cluster_idx = 0
        self.videos_to_delete = set()
        self.photo_references = []  # To prevent garbage collection of images
        self.base_directory = detector.directory  # Store the base directory for relative path calculation
        
        # Main frame
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Navigation frame
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(fill=tk.X, pady=5)
        
        # Group navigation
        self.prev_btn = ttk.Button(self.nav_frame, text="Previous Group", command=self.previous_group)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.group_label = ttk.Label(self.nav_frame, text="Group 0/0", font=("Arial", 12, "bold"))
        self.group_label.pack(side=tk.LEFT, padx=20)
        
        self.next_btn = ttk.Button(self.nav_frame, text="Next Group", command=self.next_group)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Delete button
        self.delete_btn = ttk.Button(self.nav_frame, text="Delete Selected Videos", command=self.delete_selected)
        self.delete_btn.pack(side=tk.RIGHT, padx=10)
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="")
        self.status_label.pack(fill=tk.X, pady=5)
        
        # Content frame with scrollbar
        self.content_frame_outer = ttk.Frame(self.main_frame)
        self.content_frame_outer.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.content_frame_outer)
        self.scrollbar = ttk.Scrollbar(self.content_frame_outer, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Content frame will be inside the canvas
        self.content_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        
        # Configure canvas scrolling
        self.content_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Bind mouse wheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Display first group
        if self.clusters:
            self.display_current_cluster()
    
    def on_canvas_configure(self, event):
        """Update the width of the window when the canvas is resized."""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def previous_group(self):
        """Navigate to the previous group of similar videos."""
        if not self.clusters:
            return
        
        self.current_cluster_idx = (self.current_cluster_idx - 1) % len(self.clusters)
        self.display_current_cluster()
    
    def next_group(self):
        """Navigate to the next group of similar videos."""
        if not self.clusters:
            return
        
        self.current_cluster_idx = (self.current_cluster_idx + 1) % len(self.clusters)
        self.display_current_cluster()
    
    def display_current_cluster(self):
        """Display the current group of similar videos with multiple thumbnails per video.
        Videos are arranged in a 2-column grid layout with proper scrolling."""
        # Clear previous widgets
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        self.photo_references.clear()  # Clear old references
        
        if not self.clusters:
            self.group_label.config(text="No similar videos found")
            self.status_label.config(text="No similar videos to display")
            return
        
        # Update group label
        cluster_idx = self.current_cluster_idx + 1  # 1-indexed for display
        self.group_label.config(text=f"Group {cluster_idx}/{len(self.clusters)}")
        
        # Get current cluster
        cluster = self.clusters[self.current_cluster_idx]
        
        # Create a unique identifier for this cluster based on the sorted paths
        # This ensures we recognize the same cluster regardless of navigation direction
        cluster_id = tuple(sorted(v.path for v in cluster))
        
        # Check if this is the first time we're viewing this cluster
        if not hasattr(self, 'visited_cluster_ids'):
            self.visited_cluster_ids = set()  # Create tracking set if it doesn't exist
        
        is_first_visit = cluster_id not in self.visited_cluster_ids
        
        # If first visit, auto-mark all videos except the highest quality one
        if is_first_visit:
            self.visited_cluster_ids.add(cluster_id)  # Mark this cluster as visited
            for i, video in enumerate(cluster):
                if i > 0:  # Skip the highest quality video (first one)
                    self.videos_to_delete.add(video.path)
        
        # Count how many are currently marked for deletion in this cluster
        marked_count = sum(1 for video in cluster if video.path in self.videos_to_delete)
        
        # Update status label
        self.status_label.config(text=f"Marked {marked_count}/{len(cluster)} videos for deletion")
        
        # Display videos in a 2-column grid layout
        videos_per_row = 2  # Always arrange 2 videos per row
        max_thumb_width = 180  # Control the width of thumbnails
        
        for i, video in enumerate(cluster):
            # Calculate row and column in grid
            row = i // videos_per_row
            col = i % videos_per_row
            
            # Create frame for each video
            video_frame = ttk.LabelFrame(self.content_frame, text=f"Video {i+1}")
            video_frame.grid(row=row, column=col, padx=10, pady=10, sticky="n")
            
            # Frame for thumbnails
            thumbnails_frame = ttk.Frame(video_frame)
            thumbnails_frame.pack(padx=5, pady=5)
            
            # Display all 3 thumbnails in a row
            for j, pil_img in enumerate(video.thumbnails):
                # Maintain aspect ratio while resizing to fit in grid
                width, height = pil_img.size
                new_width = max_thumb_width
                new_height = int((height/width) * new_width)  # Keep aspect ratio
                resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert PIL image to PhotoImage
                photo = ImageTk.PhotoImage(resized_img)
                self.photo_references.append(photo)  # Keep a reference
                
                # Create label for thumbnail with position indicator
                pos_text = ["10%", "50%", "90%"][j]
                thumb_frame = ttk.LabelFrame(thumbnails_frame, text=f"Frame at {pos_text}")
                thumb_frame.grid(row=0, column=j, padx=3, pady=3)
                
                # Add thumbnail to label
                thumb_label = ttk.Label(thumb_frame, image=photo)
                thumb_label.pack()
            
            # Video information
            # Get relative path from base directory
            rel_path = os.path.relpath(os.path.dirname(video.path), self.base_directory)
            if rel_path == ".":
                rel_path = "<Root Directory>"
                
            info_text = f"Name: {os.path.basename(video.path)}\n"
            info_text += f"Path: {rel_path}\n"  # Add relative path
            info_text += f"Duration: {video.duration:.2f}s\n"
            info_text += f"Resolution: {video.resolution}\n"
            info_text += f"Size: {video.format_size()}\n"
            info_text += f"Bitrate: {video.bitrate/1000:.1f} Kbps"
            
            # Quality indicator
            quality_text = "Highest Quality" if i == 0 else f"Lower Quality (-{self.get_quality_diff(cluster[0], video):.1f}%)"
            
            # Info label
            info_label = ttk.Label(video_frame, text=info_text)
            info_label.pack(pady=5, fill=tk.X)
            
            # Quality label
            quality_label = ttk.Label(video_frame, text=quality_text, font=("Arial", 10, "bold"))
            quality_label.pack(pady=2)
            
            # Checkbox for selection - now enabled for ALL videos
            delete_var = tk.BooleanVar(value=video.path in self.videos_to_delete)
            delete_check = ttk.Checkbutton(video_frame, text="Mark for deletion", variable=delete_var, 
                                        command=lambda v=video, var=delete_var: self.toggle_delete(v, var))
            delete_check.pack(pady=5)
            
            # Button to open video
            open_btn = ttk.Button(video_frame, text="Open Video", 
                               command=lambda path=video.path: self.open_video(path))
            open_btn.pack(pady=5)
    
    def get_quality_diff(self, best_video, current_video):
        """Calculate quality difference in percentage."""
        best_quality = self.detector.get_video_quality(best_video)
        current_quality = self.detector.get_video_quality(current_video)
        if best_quality > 0:
            return (best_quality - current_quality) / best_quality * 100
        return 0.0
        
    def display_current_cluster(self):
        """Display the current group of similar videos with multiple thumbnails per video.
        Videos are arranged in a 2-column grid layout with proper scrolling."""
        # Clear previous widgets
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        self.photo_references.clear()  # Clear old references
        
        if not self.clusters:
            self.group_label.config(text="No similar videos found")
            self.status_label.config(text="No similar videos to display")
            return
        
        # Update group label
        cluster_idx = self.current_cluster_idx + 1  # 1-indexed for display
        self.group_label.config(text=f"Group {cluster_idx}/{len(self.clusters)}")
        
        # Get current cluster
        cluster = self.clusters[self.current_cluster_idx]
        
        # Auto-mark all lower quality videos for deletion
        # First, unmark any previously marked videos in this cluster
        for video in cluster:
            if video.path in self.videos_to_delete:
                self.videos_to_delete.remove(video.path)
        
        # Then mark all videos except the highest quality one for deletion
        for i, video in enumerate(cluster):
            if i > 0:  # Skip the highest quality video (first one)
                self.videos_to_delete.add(video.path)
        
        # Update status label
        self.status_label.config(text=f"Displaying {len(cluster)} similar videos (Auto-marked {len(cluster)-1} for deletion)")
        
            # Display videos in a 2-column grid layout
        videos_per_row = 2  # Always arrange 2 videos per row
        max_thumb_width = 180  # Control the width of thumbnails
        
        for i, video in enumerate(cluster):
            # Calculate row and column in grid
            row = i // videos_per_row
            col = i % videos_per_row
            
            # Create frame for each video
            video_frame = ttk.LabelFrame(self.content_frame, text=f"Video {i+1}")
            video_frame.grid(row=row, column=col, padx=10, pady=10, sticky="n")
            
                # Frame for thumbnails
            thumbnails_frame = ttk.Frame(video_frame)
            thumbnails_frame.pack(padx=5, pady=5)
            
            # Display all 3 thumbnails in a row
            for j, pil_img in enumerate(video.thumbnails):
                # Maintain aspect ratio while resizing to fit in grid
                width, height = pil_img.size
                new_width = max_thumb_width
                new_height = int((height/width) * new_width)  # Keep aspect ratio
                resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert PIL image to PhotoImage
                photo = ImageTk.PhotoImage(resized_img)
                self.photo_references.append(photo)  # Keep a reference

                # Create label for thumbnail with position indicator
                pos_text = ["10%", "50%", "90%"][j]
                thumb_frame = ttk.LabelFrame(thumbnails_frame, text=f"Frame at {pos_text}")
                thumb_frame.grid(row=0, column=j, padx=3, pady=3)
                
                # Add thumbnail to label
                thumb_label = ttk.Label(thumb_frame, image=photo)
                thumb_label.pack()
            
            # Video information
            # Get relative path from base directory
            rel_path = os.path.relpath(os.path.dirname(video.path), self.base_directory)
            if rel_path == ".":
                rel_path = "<Root Directory>"
                
            info_text = f"Name: {os.path.basename(video.path)}\n"
            info_text += f"Path: {rel_path}\n"  # Add relative path
            info_text += f"Duration: {video.duration:.2f}s\n"
            info_text += f"Resolution: {video.resolution}\n"
            info_text += f"Size: {video.format_size()}\n"
            info_text += f"Bitrate: {video.bitrate/1000:.1f} Kbps"
            
            # Quality indicator
            quality_text = "Highest Quality" if i == 0 else f"Lower Quality (-{self.get_quality_diff(cluster[0], video):.1f}%)"
            
            # Info label
            info_label = ttk.Label(video_frame, text=info_text)
            info_label.pack(pady=5, fill=tk.X)

            # Quality label
            quality_label = ttk.Label(video_frame, text=quality_text, font=("Arial", 10, "bold"))
            quality_label.pack(pady=2)
            
            # Checkbox for selection - enabled for ALL videos
            delete_var = tk.BooleanVar(value=video.path in self.videos_to_delete)
            delete_check = ttk.Checkbutton(video_frame, text="Mark for deletion", variable=delete_var,
                                         command=lambda v=video, var=delete_var: self.toggle_delete(v, var))
            delete_check.pack(pady=5)

            # Button to open video
            open_btn = ttk.Button(video_frame, text="Open Video", 
                               command=lambda path=video.path: self.open_video(path))
            open_btn.pack(pady=5)

    def get_quality_diff(self, best_video, current_video):
        """Calculate quality difference in percentage."""
        best_quality = self.detector.get_video_quality(best_video)
        current_quality = self.detector.get_video_quality(current_video)
        if best_quality > 0:
            return (best_quality - current_quality) / best_quality * 100
        return 0.0

    def toggle_delete(self, video, checkbox_var):
        """Mark or unmark a video for deletion.
        Ensures at least one video remains unmarked in each cluster."""
        current_cluster = self.clusters[self.current_cluster_idx]
        
        # Count how many videos would be marked after this toggle
        would_be_marked = set(self.videos_to_delete)
        
        if checkbox_var.get():  # If being checked
            would_be_marked.add(video.path)
        else:  # If being unchecked
            would_be_marked.discard(video.path)
        
        # Count how many videos in current cluster would be marked
        cluster_videos_paths = {v.path for v in current_cluster}
        marked_in_cluster = len(would_be_marked.intersection(cluster_videos_paths))
        
        # Check if this would mark all videos in current cluster
        if marked_in_cluster == len(current_cluster):
            messagebox.showwarning("Warning", "At least one video must remain unmarked in each group!")
            checkbox_var.set(False)  # Revert the checkbox
            return
        
        # Apply the change since it's valid
        if checkbox_var.get():
            self.videos_to_delete.add(video.path)
        else:
            self.videos_to_delete.discard(video.path)
        
        # Update the status label
        marked_count = sum(1 for v in current_cluster if v.path in self.videos_to_delete)
        total_count = len(current_cluster)
        self.status_label.config(text=f"Marked {marked_count}/{total_count} videos for deletion")
    
    def delete_selected(self):
        """Delete selected videos with confirmation."""
        if not self.videos_to_delete:
            messagebox.showinfo("Info", "No videos selected for deletion")
            return
        
        # Confirmation dialog
        count = len(self.videos_to_delete)
        confirm = messagebox.askyesno("Confirm Deletion", 
                                    f"Are you sure you want to delete {count} videos?")
        
        if confirm:
            deleted = 0
            errors = 0
            
            for path in list(self.videos_to_delete):  # Use list to avoid modifying during iteration
                try:
                    os.remove(path)
                    deleted += 1
                except Exception as e:
                    print(f"Error deleting {path}: {str(e)}")
                    errors += 1
            
            # Update status
            self.status_label.config(text=f"Deleted {deleted} videos. Errors: {errors}")
            
            # Remove deleted videos from clusters
            self.update_clusters_after_deletion()
            
            # Clear selection
            self.videos_to_delete.clear()
            
            # Refresh display
            self.display_current_cluster()
    
    def update_clusters_after_deletion(self):
        """Remove deleted videos from clusters and update the UI."""
        # Remove empty clusters and deleted videos
        for i in range(len(self.clusters) - 1, -1, -1):  # Iterate backwards
            # Remove deleted videos from this cluster
            self.clusters[i] = [v for v in self.clusters[i] if os.path.exists(v.path)]
            
            # If cluster is now empty or has only one video, remove it
            if len(self.clusters[i]) <= 1:
                self.clusters.pop(i)
        
        # Update current cluster index if needed
        if not self.clusters:
            self.current_cluster_idx = 0
            self.group_label.config(text="No similar videos found")
            self.status_label.config(text="All similar video groups have been processed")
        else:
            if self.current_cluster_idx >= len(self.clusters):
                self.current_cluster_idx = len(self.clusters) - 1

def select_directory():
    """Open a file dialog to select a directory."""
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select Directory with Videos")
    
    # Don't destroy the root yet as we need it for the parameters dialog
    if directory:  # Only if directory was selected
        root.deiconify()  # Make root visible again for our dialog
        return directory, root
    else:
        root.destroy()
        return None, None


def show_parameter_dialog(root, initial_threshold=0.85, initial_tolerance=1.0):
    """Show a dialog for adjusting detection parameters with intuitive sliders."""
    params_dialog = tk.Toplevel(root)
    params_dialog.title("Adjust Detection Parameters")
    params_dialog.geometry("800x850")  # Make window larger to fit more controls
    params_dialog.minsize(800, 850)   # Set minimum size
    params_dialog.resizable(True, True)  # Allow resizing
    params_dialog.grab_set()  # Make dialog modal
    
    # Set icon and style
    style = ttk.Style(params_dialog)
    style.configure("TLabel", font=("Arial", 10))
    style.configure("Header.TLabel", font=("Arial", 12, "bold"))
    style.configure("SubHeader.TLabel", font=("Arial", 11, "bold"))
    style.configure("Desc.TLabel", font=("Arial", 9))
    style.configure("TButton", font=("Arial", 10, "bold"))
    style.configure("TCheckbutton", font=("Arial", 10))
    
    # Use a master container with proper layout
    container = ttk.Frame(params_dialog)
    container.pack(fill=tk.BOTH, expand=True)
    
    # Scrollable canvas for the settings
    canvas = tk.Canvas(container)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Enable mousewheel scrolling
    def _on_mousewheel(event):
        # Check if mouse is over the canvas
        x, y = params_dialog.winfo_pointerxy()
        widget_under_mouse = params_dialog.winfo_containing(x, y)
        if widget_under_mouse and (widget_under_mouse == canvas or widget_under_mouse.master == scrollable_frame or widget_under_mouse.master.master == scrollable_frame):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Fixed button container at the top of the window (not in scrollable area)
    btn_container = ttk.Frame(params_dialog, padding="10 10 10 10")
    btn_container.pack(side=tk.TOP, fill=tk.X)
    
    # Header
    header = ttk.Label(btn_container, text="Detection Parameters", style="Header.TLabel")
    header.pack(pady=(0, 10))
    
    def on_ok():
        results["similarity_threshold"] = similarity_threshold.get()
        results["duration_tolerance"] = duration_tolerance.get()
        results["hash_size"] = hash_size.get()
        results["frame_sampling_density"] = frame_sampling_density.get()
        results["max_duration_diff"] = max_duration_diff.get()
        results["enable_partial_match"] = enable_partial_match.get()
        params_dialog.destroy()
    
    def on_cancel():
        results["cancelled"] = True
        params_dialog.destroy()
    
    # Style for bigger buttons
    style.configure("Big.TButton", font=("Arial", 12, "bold"), padding=10)
    
    # OK button - large, prominent, and centered
    ok_btn = ttk.Button(btn_container, text="RUN WITH THESE SETTINGS", command=on_ok, style="Big.TButton")
    ok_btn.pack(side=tk.TOP, pady=10, ipadx=20, ipady=10, fill=tk.X)  # Add internal padding to make button bigger
    
    # Cancel button
    cancel_btn = ttk.Button(btn_container, text="Cancel", command=on_cancel)
    cancel_btn.pack(side=tk.TOP, pady=(0, 10))
    
    # Add a separator for visual clarity
    ttk.Separator(params_dialog, orient='horizontal').pack(fill=tk.X, side=tk.TOP)
    
    # Main frame with padding inside the scrollable area
    main_frame = ttk.Frame(scrollable_frame, padding="20 20 20 20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    description = ttk.Label(main_frame, text="Adjust the parameters below to control how videos are detected and compared. \nSliding toward 'Detect More Videos' will make the detector more sensitive but may include false matches.", style="Desc.TLabel", wraplength=550)
    description.pack(pady=(0, 20))
    
    # Parameter variables
    similarity_threshold = tk.DoubleVar(value=initial_threshold)
    duration_tolerance = tk.DoubleVar(value=initial_tolerance)
    hash_size = tk.IntVar(value=8)
    frame_sampling_density = tk.IntVar(value=3)  # Number of sample frames
    max_duration_diff = tk.DoubleVar(value=3.0)  # Default 3 seconds
    enable_partial_match = tk.BooleanVar(value=True)
    
    # Create frames for each parameter slider
    def create_slider_frame(parent, text, variable, from_, to, resolution, left_label, right_label):
        frame = ttk.LabelFrame(parent, text=text, padding="10 10 10 10")
        frame.pack(fill=tk.X, pady=10)
        
        # Labels for slider ends
        labels_frame = ttk.Frame(frame)
        labels_frame.pack(fill=tk.X, pady=5)
        
        left = ttk.Label(labels_frame, text=left_label, style="Desc.TLabel")
        left.pack(side=tk.LEFT)
        
        right = ttk.Label(labels_frame, text=right_label, style="Desc.TLabel")
        right.pack(side=tk.RIGHT)
        
        # Slider
        slider = ttk.Scale(frame, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL, length=550)
        slider.pack(fill=tk.X)
        
        # Current value label
        current_val_frame = ttk.Frame(frame)
        current_val_frame.pack(fill=tk.X, pady=5)
        
        current_val = ttk.Label(current_val_frame, text=f"Current value: {variable.get():.2f}")
        current_val.pack(side=tk.RIGHT)
        
        # Update label when slider moves
        def update_label(event=None):
            if isinstance(variable.get(), float):
                current_val.configure(text=f"Current value: {variable.get():.2f}")
            else:
                current_val.configure(text=f"Current value: {variable.get()}")
                
        slider.bind("<Motion>", update_label)
        slider.bind("<ButtonRelease-1>", update_label)
        
        return frame
    
    # Create section headers
    basic_section = ttk.Label(main_frame, text="Basic Detection Parameters", style="SubHeader.TLabel")
    basic_section.pack(pady=(0, 10), anchor="w")
    
    # Create the basic sliders
    create_slider_frame(
        main_frame, 
        "Similarity Threshold", 
        similarity_threshold, 
        0.6, 0.95, 0.01,
        "Detect More Videos (less strict)",
        "Detect Fewer Videos (more strict)"
    )
    
    create_slider_frame(
        main_frame, 
        "Duration Tolerance (seconds)", 
        duration_tolerance, 
        0.5, 3.0, 0.1,
        "Detect More Videos (larger tolerance)",
        "Detect Fewer Videos (smaller tolerance)"
    )
    
    # Advanced section
    advanced_section = ttk.Label(main_frame, text="Advanced Detection Parameters", style="SubHeader.TLabel")
    advanced_section.pack(pady=(20, 10), anchor="w")
    
    create_slider_frame(
        main_frame, 
        "Hash Size (affects comparison precision)", 
        hash_size, 
        4, 16, 1,
        "Faster Processing (less precise)",
        "Slower Processing (more precise)"
    )
    
    create_slider_frame(
        main_frame, 
        "Frame Sampling (number of frames to compare)", 
        frame_sampling_density, 
        2, 5, 1,
        "Faster Processing (fewer frames)",
        "Slower Processing (more frames)"
    )
    
    # Partial matching section
    partial_section = ttk.Label(main_frame, text="Partial Content Matching", style="SubHeader.TLabel")
    partial_section.pack(pady=(20, 10), anchor="w")
    
    # Enable/disable partial matching
    enable_frame = ttk.Frame(main_frame)
    enable_frame.pack(fill=tk.X, pady=5)
    
    enable_check = ttk.Checkbutton(
        enable_frame, 
        text="Enable partial content matching (find similar videos with different durations)", 
        variable=enable_partial_match
    )
    enable_check.pack(anchor="w", padx=10)
    
    # Explanation
    explanation = ttk.Label(
        enable_frame, 
        text="When enabled, this will find cases where a shorter video matches part of a longer video", 
        style="Desc.TLabel"
    )
    explanation.pack(anchor="w", padx=30, pady=(0, 10))
    
    # Max duration difference slider
    create_slider_frame(
        main_frame, 
        "Maximum Duration Difference (seconds)", 
        max_duration_diff, 
        5.0, 60.0, 1.0,
        "Small Difference (faster)",
        "Large Difference (slower)"
    )
    
    # Results variable to return
    results = {}
    
    # Make the dialog modal
    params_dialog.protocol("WM_DELETE_WINDOW", on_cancel)  # Handle window close button
    params_dialog.transient(root)
    params_dialog.wait_window(params_dialog)
    
    return results


def main():
    """Main function to run the similar videos detector."""
    parser = argparse.ArgumentParser(description="Detect similar videos in a directory and its subdirectories.")
    parser.add_argument("--dir", type=str, help="Directory to scan for videos")
    parser.add_argument("--gui", action="store_true", help="Use GUI for directory selection")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0-1)")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Duration tolerance in seconds")
    
    args = parser.parse_args()
    
    directory = args.dir
    root = None
    
    if args.gui or not directory:
        directory, root = select_directory()
    
    if not directory:
        print("No directory selected. Exiting.")
        return
    
    # If we need to create a root window and didn't get one from select_directory
    if root is None:
        root = tk.Tk()
        root.withdraw()
    
    # Show parameter dialog
    params = show_parameter_dialog(root, initial_threshold=args.threshold, initial_tolerance=args.tolerance)
    
    # Check if cancelled
    if params.get("cancelled", False):
        print("Parameter adjustment cancelled. Exiting.")
        return
    
    # Apply parameters
    detector = SimilarVideosDetector(
        directory=directory,
        similarity_threshold=params.get("similarity_threshold", args.threshold),
        duration_tolerance=params.get("duration_tolerance", args.tolerance),
        hash_size=params.get("hash_size", 8),
        frame_sampling_density=params.get("frame_sampling_density", 3),
        max_duration_diff=params.get("max_duration_diff", 10.0),
        enable_partial_match=params.get("enable_partial_match", True)
    )
    
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
    if root.winfo_exists():
        root.deiconify()
    else:
        root = tk.Tk()
    
    root.title("Similar Videos Detector")
    root.geometry("1200x800")
    app = VideoComparisonGUI(root, detector)
    root.mainloop()
    
    # Clean up temporary files when closing
    try:
        for video in detector.videos:
            for path in video.thumbnail_paths:
                if os.path.exists(path):
                    os.remove(path)
    except Exception as e:
        print(f"Error cleaning up temporary files: {str(e)}")


if __name__ == "__main__":
    main()
