# Similar Videos Detector

This tool helps you find similar videos across directories and subdirectories, even when they have different file sizes or bitrates. It analyzes video duration and content to identify potential duplicates.

## Features

- Recursively scans directories for video files
- Groups videos by similar duration
- Analyzes visual content to detect similar videos
- Generates detailed reports with recommendations on which videos to keep or delete
- Sorts similar videos by quality (resolution, bitrate, and file size)

## Requirements

Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

```
python similar_videos_detector.py [directory] [options]
```

### Arguments

- `directory`: The directory to scan for videos (will include all subdirectories)

### Options

- `--threshold`: Similarity threshold (0.0-1.0, default: 0.85)
- `--duration-tolerance`: Duration tolerance in seconds (default: 1.0)
- `--report`: Custom output report file path (default: auto-generated)

### Example

```
python similar_videos_detector.py "C:\Videos" --threshold 0.80 --duration-tolerance 2.0
```

## How it works

1. The script first scans the specified directory and all subdirectories for video files
2. It extracts duration, resolution, bitrate, and file size information from each video
3. Videos are grouped by similar duration (within the specified tolerance)
4. For each group, videos are compared using visual fingerprinting:
   - Sample frames are extracted from each video at regular intervals
   - Visual fingerprints are generated from these frames
   - Videos with similar fingerprints above the threshold are marked as similar
5. Similar videos are grouped into clusters
6. For each cluster, videos are sorted by quality (resolution, bitrate, then size)
7. A report is generated with recommendations on which videos to keep (highest quality) and which could be deleted

## Report Format

The generated report includes:

- A list of similar video groups
- For each group:
  - The highest quality video (recommended to keep)
  - Lower quality versions of the same content (potential deletions)
  - Quality difference percentage
  - File paths
- Potential space savings if all lower quality duplicates are removed

## Notes

- **Important**: This tool only provides recommendations. Always review the report and verify the videos before deleting anything.
- The script will not delete any files automatically; it only generates a report.
- Higher threshold values (closer to 1.0) mean stricter matching, which may miss some similar videos but will have fewer false positives.
- Lower threshold values will find more potential matches but may include some false positives.
