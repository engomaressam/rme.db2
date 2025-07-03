import os
import sys
import cv2
import numpy as np
import hashlib
import argparse
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from moviepy.editor import VideoFileClip


class VideoInfo:
    """Store video information and comparison results."""
    def __init__(self, path, duration=None, resolution=None, size=None, bitrate=None):
        self.path = path
        self.duration = duration
        self.resolution = resolution
        self.size = size
        self.bitrate = bitrate
        self.fingerprint = None

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
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Resize frame to a small size for faster processing
                small_frame = cv2.resize(frame, (32, 32))
                # Convert to grayscale
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                # Compute average hash
                avg_val = gray.mean()
                binary = (gray > avg_val).flatten()
                hash_value = ''.join(['1' if b else '0' for b in binary])
                frame_hashes.append(hash_value)
            
            cap.release()
            
            # Combine hashes into a fingerprint
            video_info.fingerprint = frame_hashes
            return frame_hashes
            
        except Exception as e:
            print(f"Error calculating fingerprint for {video_info.path}: {str(e)}")
            return None
    
    def compare_fingerprints(self, fingerprint1, fingerprint2):
        """Compare two video fingerprints and return similarity score (0-1)."""
        if not fingerprint1 or not fingerprint2 or len(fingerprint1) != len(fingerprint2):
            return 0.0
        
        matches = 0
        total = len(fingerprint1) * len(fingerprint1[0])
        
        for h1, h2 in zip(fingerprint1, fingerprint2):
            for bit1, bit2 in zip(h1, h2):
                if bit1 == bit2:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
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
        return similar_videos
    
    def generate_report(self, output_file=None):
        """Generate a report of similar videos with recommendations."""
        if not self.similar_videos:
            print("No similar videos found.")
            return
            
        # Group by clusters (transitive similarity)
        clusters = self._cluster_similar_videos()
        
        # If no output file specified, use timestamp
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     f"similar_videos_report_{timestamp}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Similar Videos Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source directory: {self.directory}\n")
            f.write(f"Processed files: {self.processed_files}\n")
            f.write(f"Similarity threshold: {self.similarity_threshold*100:.1f}%\n\n")
            
            f.write(f"Found {len(self.similar_videos)} similar video pairs in {len(clusters)} groups\n\n")
            
            for i, cluster in enumerate(clusters, 1):
                f.write(f"\nSimilar Video Group #{i} ({len(cluster)} videos):\n")
                f.write("-" * 80 + "\n")
                
                # Sort videos by quality (resolution, then bitrate, then size)
                sorted_videos = sorted(cluster, key=lambda v: (-v.resolution[0] * v.resolution[1], -v.bitrate, -v.size))
                
                # The first video is the highest quality
                best_video = sorted_videos[0]
                f.write(f"✓ KEEP: {best_video}\n")
                f.write(f"  Path: {best_video.path}\n\n")
                
                # List the rest as potential deletions
                for video in sorted_videos[1:]:
                    quality_diff = self._compare_quality(best_video, video)
                    f.write(f"× POTENTIAL DELETE: {video}\n")
                    f.write(f"  Path: {video.path}\n")
                    f.write(f"  Quality: {quality_diff}% lower than best video\n\n")
                
                f.write("\n")
            
            # Space savings estimate
            potential_savings = sum(v.size for c in clusters for v in c[1:])
            f.write(f"\nPotential space savings: {self._format_size(potential_savings)}\n")
        
        print(f"Report generated: {output_file}")
        return output_file
    
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

def main():
    parser = argparse.ArgumentParser(description='Detect similar videos in a directory and subdirectories')
    parser.add_argument('--directory', help='Directory to scan for videos (if not provided, a folder browser will appear)')
    parser.add_argument('--threshold', type=float, default=0.85, 
                        help='Similarity threshold (0.0-1.0, default: 0.85)')
    parser.add_argument('--duration-tolerance', type=float, default=1.0,
                        help='Duration tolerance in seconds (default: 1.0)')
    parser.add_argument('--report', type=str, default=None,
                        help='Output report file path (default: auto-generated)')
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
    
    detector = SimilarVideosDetector(
        directory=directory,
        similarity_threshold=args.threshold,
        duration_tolerance=args.duration_tolerance
    )
    
    detector.find_video_files()
    if not detector.videos:
        print("No video files found in the specified directory.")
        return 0
        
    detector.extract_video_info()
    similar_videos = detector.find_similar_videos()
    
    if not similar_videos:
        print("No similar videos found.")
        return 0
        
    report_file = detector.generate_report(args.report)
    print(f"\nFound {len(similar_videos)} similar video pairs.")
    print(f"Check the report at: {report_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
