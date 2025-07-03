import os
import sys
import cv2
import numpy as np
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

# ML-specific imports
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

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
    """Detect similar videos using a pre-trained neural network."""
    def __init__(self, directory, extensions=None, similarity_threshold=0.95, duration_tolerance=1.0):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model and print information about it
        print(f"\nUsing device: {self.device}")
        print("Loading ResNet-50 model...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        print(f"Model loaded: ResNet-50 (ImageNet1K-V1 weights)")
        self.model.eval()
        
        # Image transformation pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # Create a directory for thumbnails if it doesn't exist
        self.thumbnail_dir = r"C:\Windows\Temp\video_thumbnails"
        os.makedirs(self.thumbnail_dir, exist_ok=True)

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

    def get_feature_vector(self, image):
        """Extracts a feature vector from a single image frame using the ML model."""
        if not isinstance(image, Image.Image):
            # Assuming image is a numpy array from OpenCV (BGR), convert to PIL Image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
        image_t = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature_vector = self.model(image_t)
        return feature_vector.squeeze()

    def calculate_video_fingerprint(self, video_info):
        """Calculate a fingerprint for the video using deep features from sample frames."""
        if video_info.fingerprint is not None:
            return video_info.fingerprint
            
        try:
            cap = cv2.VideoCapture(video_info.path)
            if not cap.isOpened():
                raise Exception("Could not open video")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise Exception("Invalid frame count")
            
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            aspect_ratio = orig_width / orig_height if orig_height > 0 else 16/9

            positions = [0.1, 0.5, 0.9]
            feature_vectors = []
            video_info.thumbnails = []
            video_info.thumbnail_paths = []
            
            for pos in positions:
                frame_idx = int(total_frames * pos)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Generate feature vector for similarity comparison
                feature_vector = self.get_feature_vector(frame)
                feature_vectors.append(feature_vector)
                
                # Save thumbnail for GUI (same as before)
                unique_id = str(uuid.uuid4())[:8]
                file_name = f"{os.path.splitext(os.path.basename(video_info.path))[0]}_{pos}_{unique_id}.jpg"
                thumbnail_path = os.path.join(TEMP_DIR, file_name)
                
                thumb_width = 320
                thumb_height = int(thumb_width / aspect_ratio)
                
                thumbnail = cv2.resize(frame, (thumb_width, thumb_height))
                cv2.imwrite(thumbnail_path, thumbnail)
                
                rgb_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_thumbnail)
                
                video_info.thumbnails.append(pil_image)
                video_info.thumbnail_paths.append(thumbnail_path)
            
            cap.release()
            
            # Store the feature vectors as the fingerprint
            video_info.fingerprint = feature_vectors
            return feature_vectors
            
        except Exception as e:
            print(f"Error calculating fingerprint for {video_info.path}: {str(e)}")
            return None

    def compare_fingerprints(self, fingerprint1, fingerprint2):
        """Compare two video fingerprints (lists of feature vectors) using cosine similarity."""
        if not fingerprint1 or not fingerprint2 or len(fingerprint1) == 0 or len(fingerprint2) == 0:
            return 0.0
        
        max_similarity = 0.0
        
        # Calculate cosine similarity between all pairs of feature vectors
        for f1 in fingerprint1:
            for f2 in fingerprint2:
                similarity = torch.nn.functional.cosine_similarity(f1, f2, dim=0)
                if similarity.item() > max_similarity:
                    max_similarity = similarity.item()
        
        return max_similarity

    def find_similar_videos(self):
        """Find similar videos by duration and visual fingerprint."""
        print("\nFinding similar videos...")
        self.similar_videos = []
        
        print("Extracting video fingerprints (using ML model)...")
        for video in tqdm(self.videos, desc="Generating Feature Vectors"):
            self.calculate_video_fingerprint(video)
        
        duration_groups = defaultdict(list)
        for video in self.videos:
            rounded_duration = round(video.duration / self.duration_tolerance) * self.duration_tolerance
            duration_groups[rounded_duration].append(video)
        
        print(f"Comparing videos in {len(duration_groups)} duration groups...")
        comparisons = 0
        
        # Using a set to avoid duplicate pairs, since comparison is symmetric
        checked_pairs = set()

        for duration, group in tqdm(duration_groups.items(), desc="Comparing duration groups"):
            # Combine current group with adjacent groups for comparison
            all_videos_to_compare = list(group)
            for d_offset in [-self.duration_tolerance, self.duration_tolerance]:
                adjacent_duration = duration + d_offset
                if adjacent_duration in duration_groups:
                    all_videos_to_compare.extend(duration_groups[adjacent_duration])

            for i in range(len(group)):
                for j in range(len(all_videos_to_compare)):
                    video1 = group[i]
                    video2 = all_videos_to_compare[j]

                    if video1.path == video2.path:
                        continue
                    
                    pair = tuple(sorted((video1.path, video2.path)))
                    if pair in checked_pairs:
                        continue
                    
                    similarity = self.compare_fingerprints(video1.fingerprint, video2.fingerprint)
                    comparisons += 1
                    checked_pairs.add(pair)

                    if similarity >= self.similarity_threshold:
                        self.similar_videos.append((video1, video2, similarity))
        
        print(f"Performed {comparisons} comparisons")
        print(f"Found {len(self.similar_videos)} similar video pairs\n")
        
        self.group_similar_videos()
        
        return self.similar_videos
    
    def group_similar_videos(self):
        """Group similar videos into clusters."""
        graph = defaultdict(set)
        for video1, video2, _ in self.similar_videos:
            graph[video1.path].add(video2.path)
            graph[video2.path].add(video1.path)
        
        visited = set()
        clusters = []
        
        for video in self.videos:
            if video.path in visited:
                continue
            
            cluster = []
            stack = [video]
            visited.add(video.path)
            
            while stack:
                current = stack.pop()
                cluster.append(current)
                
                for neighbor_path in graph[current.path]:
                    if neighbor_path not in visited:
                        neighbor = next((v for v in self.videos if v.path == neighbor_path), None)
                        if neighbor:
                            stack.append(neighbor)
                            visited.add(neighbor_path)
            
            cluster.sort(key=self.get_video_quality, reverse=True)
            
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


def show_settings_dialog(parent):
    """Create a simplified settings dialog that won't get stuck"""
    # Create a result container with default values
    result = {
        'threshold': 0.95,
        'tolerance': 2.0
    }
    
    # Create our dialog as a new Toplevel window
    settings_win = tk.Toplevel(parent)
    settings_win.title("Adjust Detection Settings")
    settings_win.geometry("450x300")
    settings_win.resizable(False, False)
    
    # Center window and make it visible
    screen_width = settings_win.winfo_screenwidth()
    screen_height = settings_win.winfo_screenheight()
    x = (screen_width // 2) - (450 // 2)
    y = (screen_height // 2) - (300 // 2)
    settings_win.geometry(f"450x300+{x}+{y}")
    
    # Make sure it stays on top and gets focus
    settings_win.attributes("-topmost", True)
    
    # Add the sliders
    threshold_var = tk.DoubleVar(value=0.95)
    tk.Label(settings_win, text="Similarity Threshold:", font=("Arial", 10, "bold")).pack(pady=(20,0))
    threshold_label = tk.Label(settings_win, text=f"{threshold_var.get():.2f}")
    threshold_label.pack()
    
    def update_threshold_label(value):
        threshold_label.config(text=f"{float(value):.2f}")
        result['threshold'] = float(value)  # Update result as slider moves
    
    tk.Scale(settings_win, from_=0.80, to=1.0, resolution=0.01, orient="horizontal", 
             variable=threshold_var, command=update_threshold_label).pack(fill="x", padx=50)
    
    tolerance_var = tk.DoubleVar(value=2.0)
    tk.Label(settings_win, text="Duration Tolerance (seconds):", font=("Arial", 10, "bold")).pack(pady=(20,0))
    tolerance_label = tk.Label(settings_win, text=f"{tolerance_var.get():.1f}s")
    tolerance_label.pack()
    
    def update_tolerance_label(value):
        tolerance_label.config(text=f"{float(value):.1f}s")
        result['tolerance'] = float(value)  # Update result as slider moves
    
    tk.Scale(settings_win, from_=0.0, to=10.0, resolution=0.1, orient="horizontal", 
             variable=tolerance_var, command=update_tolerance_label).pack(fill="x", padx=50)
    
    # Define what happens when buttons are pressed
    def on_apply():
        print("Settings applied with:")
        print(f" - Threshold: {result['threshold']}")
        print(f" - Tolerance: {result['tolerance']}")
        settings_win.destroy()
    
    def on_cancel():
        # Reset to defaults
        result.clear()
        print("Settings cancelled")
        settings_win.destroy()
    
    # Add the buttons
    button_frame = tk.Frame(settings_win)
    button_frame.pack(pady=20)
    tk.Button(button_frame, text="Apply", command=on_apply, width=10).pack(side="left", padx=10)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side="left", padx=10)
    
    # This is a simpler way to wait for interaction
    settings_win.focus_set()
    parent.wait_window(settings_win)
    return result

def choose_mode():
    """Show a simple dialog to let the user choose between simple and advanced modes"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Create a small dialog
    dialog = tk.Toplevel(root)
    dialog.title("Choose Mode")
    dialog.geometry("300x150")
    dialog.resizable(False, False)
    
    # Center the dialog
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    dialog.attributes("-topmost", True)
    dialog.grab_set()
    dialog.focus_force()
    
    # Prevent the X button from closing it
    def disable_close():
        pass
    dialog.protocol("WM_DELETE_WINDOW", disable_close)
    
    # Instructions
    tk.Label(dialog, text="Choose Detection Mode:", font=("Arial", 12)).pack(pady=(20, 15))
    
    # Variable to store the selection
    choice = [True]  # Using a list to make it mutable in the nested functions
    
    # Button functions
    def select_simple():
        choice[0] = True  # Use default settings
        dialog.destroy()
    
    def select_advanced():
        choice[0] = False  # Show settings dialog
        dialog.destroy()
    
    # Buttons
    button_frame = tk.Frame(dialog)
    button_frame.pack()
    tk.Button(button_frame, text="Simple Mode", command=select_simple, 
              width=15).pack(side="left", padx=5)
    tk.Button(button_frame, text="Advanced Mode", command=select_advanced, 
              width=15).pack(side="left", padx=5)
    
    # Wait for the dialog to close
    root.wait_window(dialog)
    return choice[0], root

def main():
    """Main entry point with choice of simple or advanced mode"""
    print("\n--------------------")
    print("Video Similarity Detector")
    print("--------------------\n")
    
    try:
        # Let user choose mode and get root window
        use_simple_mode, root = choose_mode()
        
        # 1. First, select the directory
        directory = filedialog.askdirectory(parent=root, title="Select the Directory Containing Your Videos")
        if not directory:
            print("No directory selected. Exiting.")
            root.destroy()
            return
        
        print(f"Selected directory: {directory}")
        
        # 2. Get settings based on chosen mode
        if use_simple_mode:
            # Simple mode: use preset values
            settings = {
                'threshold': 0.95,  # Default similarity threshold 
                'tolerance': 2.0    # Default duration tolerance (seconds)
            }
            print("\nUsing default settings:")
            print(f"- Similarity Threshold: {settings['threshold']}")
            print(f"- Duration Tolerance: {settings['tolerance']}s")
        else:
            # Advanced mode: show settings dialog
            print("\nOpening settings dialog...")
            settings = show_settings_dialog(root)
            if not settings:
                print("Settings canceled. Exiting.")
                root.destroy()
                return
            print(f"\nUsing custom settings:")
            print(f"- Similarity Threshold: {settings['threshold']}")
            print(f"- Duration Tolerance: {settings['tolerance']}s")
        
        print("\nProcessing videos... This may take some time depending on the number of videos and your GPU.")
        
        # 3. Run the detection process
        detector = SimilarVideosDetector(
            directory=directory,
            similarity_threshold=settings['threshold'],
            duration_tolerance=settings['tolerance']
        )
        
        print("Scanning directory for video files...")
        video_files = detector.scan_directory()
        if not video_files:
            print("No video files found in the selected directory.")
            messagebox.showinfo("Info", "No video files found in the selected directory.")
            root.destroy()
            return
        
        print(f"Found {len(video_files)} video files")
        
        print("Extracting video information...")
        detector.extract_video_info(video_files)
        if not detector.videos:
            print("Could not process any video files.")
            messagebox.showinfo("Info", "Could not process any video files.")
            root.destroy()
            return
        
        print(f"Processed {len(detector.videos)} videos successfully")
        
        print("Finding similar videos...")
        detector.find_similar_videos()
        if not detector.video_clusters:
            print("No similar videos found based on the current settings.")
            messagebox.showinfo("Info", "No similar videos found based on the current settings.")
            root.destroy()
            return
        
        print(f"Found {len(detector.video_clusters)} clusters of similar videos")
        
        print("Saving report...")
        detector.save_report()
        
        # 4. Launch the main comparison GUI
        print("Preparing review GUI...")
        root.deiconify()
        root.title("Similar Videos Review")
        root.geometry("1200x800")
        
        # Center the main window
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        print("\nLaunching comparison GUI...")
        app = VideoComparisonGUI(root, detector)
        
        # Set up cleanup for when the window is closed
        def on_closing():
            print("Cleaning up temporary files...")
            try:
                for video in detector.videos:
                    for path in video.thumbnail_paths:
                        if os.path.exists(path):
                            os.remove(path)
            except Exception as e:
                print(f"Error cleaning up temporary files: {str(e)}")
            root.destroy()
            print("Application terminated.")
            
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"An error occurred: {str(e)}\n\nCheck the console for details.")
        
    print("\n--------------------")
    print("Application terminated")
    print("--------------------")

if __name__ == "__main__":
    main()
