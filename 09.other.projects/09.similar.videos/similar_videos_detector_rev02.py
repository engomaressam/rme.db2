import os
import sys
import cv2
import numpy as np
import hashlib
import argparse
import json
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
import shutil
from moviepy.video.io.VideoFileClip import VideoFileClip


class VideoInfo:
    """Store video information and comparison results."""
    def __init__(self, path, duration=None, resolution=None, size=None, bitrate=None):
        self.path = path
        self.duration = duration
        self.resolution = resolution
        self.size = size
        self.bitrate = bitrate
        self.fingerprint = None
        self.thumbnail = None  # Added for GUI display

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
        self.duration_tolerance = duration_tolerance  # in seconds
        self.videos = []
        self.similar_videos = []
        self.duration_groups = defaultdict(list)
        self.processed_files = 0
        self.skipped_files = 0
        self.video_clusters = []  # For storing clustered similar videos

    def find_video_files(self):
        """Find all video files in the directory and subdirectories."""
        print(f"Scanning {self.directory} for video files...")
        for root, _, files in os.walk(self.directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.extensions):
                    video_path = os.path.join(root, file)
                    self.videos.append(video_path)
        print(f"Found {len(self.videos)} video files")
        return self.videos
    
    def extract_video_info(self):
        """Extract information from video files including duration."""
        print("Extracting video information...")
        valid_videos = []
        for video_path in tqdm(self.videos):
            try:
                clip = VideoFileClip(video_path)
                file_size = os.path.getsize(video_path)
                
                # Calculate bitrate
                bitrate = (file_size * 8) / clip.duration if clip.duration > 0 else 0
                
                # Create VideoInfo object
                video_info = VideoInfo(
                    path=video_path,
                    duration=clip.duration,
                    resolution=(clip.w, clip.h),
                    size=file_size,
                    bitrate=bitrate
                )
                
                # Group by rounded duration to handle slight differences
                # Round to nearest second
                rounded_duration = round(clip.duration)
                self.duration_groups[rounded_duration].append(video_info)
                valid_videos.append(video_info)
                
                clip.close()
                self.processed_files += 1
            except Exception as e:
                print(f"\nError processing {video_path}: {str(e)}")
                self.skipped_files += 1
        
        self.videos = valid_videos
        print(f"Successfully extracted info from {self.processed_files} videos")
        if self.skipped_files > 0:
            print(f"Skipped {self.skipped_files} files due to errors")
        return self.videos
    
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
    
    def calculate_video_fingerprint(self, video_info, num_frames=10):
        """Calculate a fingerprint for the video based on sample frames."""
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
            
            # Sample frames at regular intervals
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            frame_hashes = []
            
            # Get a thumbnail for GUI display (middle frame)
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, middle_frame = cap.read()
            if ret:
                # Resize for thumbnail
                video_info.thumbnail = cv2.resize(middle_frame, (320, 180))
                # Convert from BGR to RGB for PIL
                video_info.thumbnail = cv2.cvtColor(video_info.thumbnail, cv2.COLOR_BGR2RGB)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Generate both types of hashes for better accuracy
                dhash_val = self.dhash(frame)
                phash_val = self.phash(frame)
                frame_hashes.append((dhash_val, phash_val))
            
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
        if not fingerprint1 or not fingerprint2 or len(fingerprint1) != len(fingerprint2):
            return 0.0
        
        # Calculate similarities for each frame pair
        similarities = []
        
        for (dhash1, phash1), (dhash2, phash2) in zip(fingerprint1, fingerprint2):
            # Max possible distance for the hash size we're using
            max_dist = 64
            
            # Calculate distances (lower is better)
            dhash_dist = self.hamming_distance(dhash1, dhash2)
            phash_dist = self.hamming_distance(phash1, phash2)
            
            # Convert to similarities (higher is better)
            dhash_sim = 1.0 - (dhash_dist / max_dist)
            phash_sim = 1.0 - (phash_dist / max_dist)
            
            # Take the average of both hash similarities
            avg_sim = (dhash_sim + phash_sim) / 2
            similarities.append(avg_sim)
        
        # Return the average similarity across all frames
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0
    
    def find_similar_videos(self):
        """Find similar videos based on duration and fingerprint matching."""
        print("Finding similar videos...")
        
        # Only process duration groups with more than one video
        duration_groups_to_check = {k: v for k, v in self.duration_groups.items() if len(v) > 1}
        
        if not duration_groups_to_check:
            print("No potential duplicates found (no videos share the same duration).")
            return []
        
        print(f"Found {sum(len(v) for v in duration_groups_to_check.values())} videos in {len(duration_groups_to_check)} duration groups to check")
        
        similar_videos = []
        groups_processed = 0
        
        for duration, videos in tqdm(duration_groups_to_check.items()):
            groups_processed += 1
            
            # Calculate fingerprints for each video in this duration group
            for video in videos:
                self.calculate_video_fingerprint(video)
            
            # Compare videos within the same duration group
            for i in range(len(videos)):
                for j in range(i+1, len(videos)):
                    video1, video2 = videos[i], videos[j]
                    
                    if video1.fingerprint and video2.fingerprint:
                        similarity = self.compare_fingerprints(video1.fingerprint, video2.fingerprint)
                        
                        if similarity >= self.similarity_threshold:
                            similar_videos.append({
                                'video1': video1,
                                'video2': video2,
                                'similarity': similarity
                            })
        
        self.similar_videos = similar_videos
        
        # Group similar videos into clusters
        self.video_clusters = self._cluster_similar_videos()
        
        return similar_videos
    
    def _cluster_similar_videos(self):
        """Group similar videos into clusters using transitive similarity."""
        # Create initial clusters from similarity pairs
        videos_in_cluster = set()
        clusters = []
        
        # Sort by similarity for consistent results
        sorted_similar = sorted(self.similar_videos, key=lambda x: (-x['similarity']))
        
        for pair in sorted_similar:
            video1, video2 = pair['video1'], pair['video2']
            
            # Check if any of these videos is already in a cluster
            existing_cluster = None
            for cluster in clusters:
                if video1 in cluster or video2 in cluster:
                    existing_cluster = cluster
                    break
            
            if existing_cluster:
                # Add both videos to the existing cluster if not already there
                if video1 not in existing_cluster:
                    existing_cluster.append(video1)
                    videos_in_cluster.add(video1)
                if video2 not in existing_cluster:
                    existing_cluster.append(video2)
                    videos_in_cluster.add(video2)
            else:
                # Create a new cluster
                new_cluster = [video1, video2]
                clusters.append(new_cluster)
                videos_in_cluster.add(video1)
                videos_in_cluster.add(video2)
        
        # Sort videos in each cluster by quality
        for i, cluster in enumerate(clusters):
            clusters[i] = sorted(cluster, key=lambda v: (-v.resolution[0] * v.resolution[1], -v.bitrate, -v.size))
        
        return clusters
    
    def _compare_quality(self, video1, video2):
        """Compare quality between two videos and return percentage difference."""
        # Calculate a quality score based on resolution and bitrate
        quality1 = (video1.resolution[0] * video1.resolution[1]) * (video1.bitrate ** 0.5)
        quality2 = (video2.resolution[0] * video2.resolution[1]) * (video2.bitrate ** 0.5)
        
        # Percentage difference (how much lower quality2 is from quality1)
        if quality1 == 0:
            return 0
        
        diff_percent = max(0, (quality1 - quality2) / quality1 * 100)
        return round(diff_percent, 1)
    
    def _format_size(self, size_bytes):
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024 or unit == 'TB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024


class VideoComparisonGUI:
    """GUI for comparing and deleting similar videos."""
    def __init__(self, master, detector):
        self.master = master
        self.detector = detector
        self.clusters = detector.video_clusters
        self.current_cluster_idx = 0
        self.videos_to_delete = set()  # Track videos marked for deletion
        self.photo_references = []  # Keep references to PhotoImage objects
        
        master.title("Similar Videos Detector - Video Comparison")
        master.geometry("1200x800")
        master.configure(bg="#f0f0f0")
        
        # Top frame for navigation and controls
        self.top_frame = ttk.Frame(master)
        self.top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Group counter label
        self.group_label = ttk.Label(
            self.top_frame, 
            text=f"Group 1/{len(self.clusters)}", 
            font=("Arial", 14, "bold")
        )
        self.group_label.pack(side=tk.LEFT, padx=5)
        
        # Navigation buttons
        self.prev_button = ttk.Button(
            self.top_frame, 
            text="Previous Group", 
            command=self.prev_group,
            state=tk.DISABLED
        )
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(
            self.top_frame, 
            text="Next Group", 
            command=self.next_group
        )
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Middle frame for video comparisons
        self.video_frame = ttk.Frame(master)
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bottom frame for action buttons
        self.bottom_frame = ttk.Frame(master)
        self.bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Delete selected button
        self.delete_button = ttk.Button(
            self.bottom_frame,
            text="Delete Selected Videos",
            command=self.delete_selected,
            style="Delete.TButton"
        )
        self.delete_button.pack(side=tk.RIGHT, padx=5)
        
        # Status message
        self.status_var = tk.StringVar()
        self.status_var.set(f"Found {len(self.clusters)} groups of similar videos")
        self.status_label = ttk.Label(
            self.bottom_frame, 
            textvariable=self.status_var,
            font=("Arial", 10)
        )
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Create a style for the delete button
        self.style = ttk.Style()
        self.style.configure("Delete.TButton", foreground="red")
        
        # Initialize with first cluster
        if self.clusters:
            self.display_current_cluster()
        else:
            self.status_var.set("No similar videos found")
    
    def display_current_cluster(self):
        """Display the current cluster of similar videos"""
        # Clear previous content
        for widget in self.video_frame.winfo_children():
            widget.destroy()
        
        # Clear image references
        self.photo_references = []
        
        # Update navigation buttons state
        self.prev_button.config(state=tk.NORMAL if self.current_cluster_idx > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_cluster_idx < len(self.clusters) - 1 else tk.DISABLED)
        
        # Update group label
        self.group_label.config(text=f"Group {self.current_cluster_idx + 1}/{len(self.clusters)}")
        
        # Get the current cluster
        cluster = self.clusters[self.current_cluster_idx]
        
        # Keep track of the video widgets in this cluster
        self.video_widgets = []
        
        # Display each video in the cluster
        for i, video in enumerate(cluster):
            frame = ttk.LabelFrame(self.video_frame, text=f"Video {i+1}")
            frame.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="nsew")
            
            # Ensure the grid cells expand properly
            self.video_frame.grid_columnconfigure(0, weight=1)
            self.video_frame.grid_columnconfigure(1, weight=1)
            self.video_frame.grid_rowconfigure(i//2, weight=1)
            
            # Video thumbnail
            if video.thumbnail is not None:
                try:
                    img = Image.fromarray(video.thumbnail)
                    photo = ImageTk.PhotoImage(image=img)
                    # Store reference to prevent garbage collection
                    self.photo_references.append(photo)
                    
                    thumbnail_label = ttk.Label(frame, image=photo)
                    thumbnail_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                except Exception as e:
                    print(f"Error creating thumbnail: {str(e)}")
                    thumbnail_label = ttk.Label(frame, text="[Error loading thumbnail]")
                    thumbnail_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            else:
                # Placeholder if no thumbnail
                thumbnail_label = ttk.Label(frame, text="[No thumbnail available]")
                thumbnail_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Video information
            info_text = f"Filename: {os.path.basename(video.path)}\n"
            info_text += f"Resolution: {video.resolution[0]}x{video.resolution[1]}\n"
            info_text += f"Size: {video.format_size()}\n"
            info_text += f"Bitrate: {video.bitrate/1000:.1f} Kbps\n"
            info_text += f"Duration: {video.duration:.2f} seconds"
            
            info_label = ttk.Label(frame, text=info_text, justify=tk.LEFT)
            info_label.pack(fill=tk.X, padx=5, pady=5)
            
            # Checkbox for deletion (only available for non-highest quality videos)
            delete_var = tk.BooleanVar(value=video in self.videos_to_delete)
            
            # First video is always the highest quality, can't be deleted
            if i == 0:
                delete_check = ttk.Checkbutton(
                    frame, 
                    text="KEEP (Highest Quality)", 
                    variable=delete_var,
                    state=tk.DISABLED
                )
            else:
                quality_diff = self.detector._compare_quality(cluster[0], video)
                delete_check = ttk.Checkbutton(
                    frame, 
                    text=f"Mark for Deletion ({quality_diff}% lower quality)", 
                    variable=delete_var,
                    command=lambda v=video, var=delete_var: self.toggle_delete(v, var)
                )
            
            delete_check.pack(anchor=tk.W, padx=5, pady=5)
            
            # Open video button
            open_button = ttk.Button(
                frame, 
                text="Open Video", 
                command=lambda path=video.path: self.open_video(path)
            )
            open_button.pack(pady=5)
            
            # Store widget references
            self.video_widgets.append({
                'video': video,
                'frame': frame,
                'delete_var': delete_var,
                'delete_check': delete_check
            })
    
    def toggle_delete(self, video, var):
        """Toggle a video for deletion"""
        if var.get():
            self.videos_to_delete.add(video)
        else:
            self.videos_to_delete.discard(video)
            
        # Update status message
        self.status_var.set(f"{len(self.videos_to_delete)} videos marked for deletion")
    
    def open_video(self, path):
        """Open the video in the default video player"""
        if os.path.exists(path):
            os.startfile(path)
        else:
            self.status_var.set(f"Error: Video file not found at {path}")
    
    def prev_group(self):
        """Navigate to the previous cluster"""
        if self.current_cluster_idx > 0:
            self.current_cluster_idx -= 1
            self.display_current_cluster()
    
    def next_group(self):
        """Navigate to the next cluster"""
        if self.current_cluster_idx < len(self.clusters) - 1:
            self.current_cluster_idx += 1
            self.display_current_cluster()
    
    def delete_selected(self):
        """Delete the videos marked for deletion"""
        if not self.videos_to_delete:
            self.status_var.set("No videos marked for deletion")
            return
            
        # Confirmation dialog
        count = len(self.videos_to_delete)
        size_bytes = sum(v.size for v in self.videos_to_delete)
        size_str = self.detector._format_size(size_bytes)
        
        if tk.messagebox.askokcancel(
            "Confirm Deletion", 
            f"Are you sure you want to delete {count} videos?\n\n"
            f"This will free up {size_str} of disk space.\n\n"
            "This operation cannot be undone!"):
            
            # Perform deletion
            deleted = 0
            failed = 0
            
            for video in self.videos_to_delete:
                try:
                    # Actually delete the file
                    os.remove(video.path)
                    deleted += 1
                except Exception as e:
                    print(f"Error deleting {video.path}: {str(e)}")
                    failed += 1
            
            # Update status
            self.status_var.set(f"Deleted {deleted} videos. Failed: {failed}")
            
            # Update the clusters by removing deleted videos
            new_clusters = []
            for cluster in self.clusters:
                # Filter out deleted videos
                new_cluster = [v for v in cluster if v not in self.videos_to_delete or not os.path.exists(v.path)]
                # Only keep clusters with at least 2 videos
                if len(new_cluster) >= 2:
                    new_clusters.append(new_cluster)
            
            # Update clusters and reset navigation
            self.clusters = new_clusters
            self.videos_to_delete.clear()
            self.current_cluster_idx = min(self.current_cluster_idx, len(self.clusters) - 1)
            
            if self.clusters:
                self.display_current_cluster()
            else:
                for widget in self.video_frame.winfo_children():
                    widget.destroy()
                self.status_var.set("All similar videos have been processed")
                self.group_label.config(text="No more similar videos")
                self.prev_button.config(state=tk.DISABLED)
                self.next_button.config(state=tk.DISABLED)


def select_directory():
    """Show directory selection dialog and return the selected path"""
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()
    
    # Show directory selection dialog
    directory = filedialog.askdirectory(
        title='Select directory containing videos',
        initialdir=os.getcwd()
    )
    
    # If user cancels, return None
    if not directory:
        return None
        
    return directory


def analyze_videos(directory, threshold, tolerance):
    """Run the video analysis process"""
    detector = SimilarVideosDetector(
        directory=directory,
        similarity_threshold=threshold,
        duration_tolerance=tolerance
    )
    
    detector.find_video_files()
    if not detector.videos:
        print("No video files found in the specified directory.")
        return None
    
    detector.extract_video_info()
    detector.find_similar_videos()
    
    if not detector.video_clusters:
        print("No similar videos found.")
        return None
    
    print(f"Found {len(detector.similar_videos)} similar video pairs in {len(detector.video_clusters)} groups")
    return detector


def main():
    parser = argparse.ArgumentParser(description='Detect, compare and delete similar videos with GUI')
    parser.add_argument('--directory', help='Directory to scan for videos (if not provided, a folder browser will appear)')
    parser.add_argument('--threshold', type=float, default=0.85, 
                        help='Similarity threshold (0.0-1.0, default: 0.85)')
    parser.add_argument('--duration-tolerance', type=float, default=1.0,
                        help='Duration tolerance in seconds (default: 1.0)')
    parser.add_argument('--gui', action='store_true', default=False,
                        help='Force open the directory selection dialog')
    args = parser.parse_args()
    
    directory = args.directory
    
    # If no directory specified or GUI flag is set, open the directory browser
    if args.gui or not directory:
        directory = select_directory()
        if not directory:  # User canceled the dialog
            print("No directory selected. Exiting.")
            return 1
    
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist or is not accessible")
        return 1
    
    # Start video analysis
    detector = analyze_videos(directory, args.threshold, args.duration_tolerance)
    
    if not detector or not detector.video_clusters:
        return 0
    
    # Create and run the GUI
    root = tk.Tk()
    app = VideoComparisonGUI(root, detector)
    root.mainloop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
