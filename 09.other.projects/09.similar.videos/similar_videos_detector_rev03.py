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
            
            # Get original aspect ratio
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            aspect_ratio = orig_width / orig_height if orig_height > 0 else 16/9  # Default if can't determine
            
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
        
        # Compare videos within same duration group and adjacent groups
        for duration, group in tqdm(duration_groups.items(), desc="Comparing duration groups"):
            # Get videos from adjacent duration groups (within duration_tolerance)
            adjacent_videos = []
            for d in [duration - self.duration_tolerance, duration + self.duration_tolerance]:
                if d in duration_groups:
                    adjacent_videos.extend(duration_groups[d])
            
            # Compare videos within the same group
            for i in range(len(group)):
                # Compare with other videos in the same group
                for j in range(i+1, len(group)):
                    similarity = self.compare_fingerprints(group[i].fingerprint, group[j].fingerprint)
                    comparisons += 1
                    if similarity >= self.similarity_threshold:
                        self.similar_videos.append((group[i], group[j], similarity))
                
                # Compare with videos in adjacent groups
                for video2 in adjacent_videos:
                    similarity = self.compare_fingerprints(group[i].fingerprint, video2.fingerprint)
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
        
        # Check if we've seen this cluster before by checking if any videos in this cluster are marked
        cluster_paths = {v.path for v in cluster}
        cluster_already_visited = any(path in self.videos_to_delete for path in cluster_paths)
        
        # Only auto-mark videos if this is the first time viewing this cluster
        if not cluster_already_visited:
            # Then mark all videos except the highest quality one for deletion
            for i, video in enumerate(cluster):
                if i > 0:  # Skip the highest quality video (first one)
                    self.videos_to_delete.add(video.path)
        
        # Count how many are currently marked for deletion
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
            info_text = f"Name: {os.path.basename(video.path)}\n"
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
            info_text = f"Name: {os.path.basename(video.path)}\n"
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
    """Open a dialog to select a directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title="Select Directory with Videos")
    root.destroy()
    return directory


def main():
    """Main function to run the similar videos detector."""
    parser = argparse.ArgumentParser(description="Detect similar videos in a directory and its subdirectories.")
    parser.add_argument("--dir", type=str, help="Directory to scan for videos")
    parser.add_argument("--gui", action="store_true", help="Use GUI for directory selection")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (0-1)")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Duration tolerance in seconds")
    
    args = parser.parse_args()
    
    directory = args.dir
    if args.gui or not directory:
        directory = select_directory()
    
    if not directory:
        print("No directory selected. Exiting.")
        return
    
    detector = SimilarVideosDetector(
        directory=directory,
        similarity_threshold=args.threshold,
        duration_tolerance=args.tolerance
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
