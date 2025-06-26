import os
import sys
from collections import defaultdict
import json
from datetime import datetime

def count_files(base_path):
    """Count files by extension and folder"""
    extension_counts = defaultdict(int)
    folder_counts = defaultdict(lambda: defaultdict(int))
    file_list = []

    try:
        for root, _, files in os.walk(base_path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    extension = os.path.splitext(file)[1].lower()
                    relative_path = os.path.relpath(root, base_path)
                    
                    # Update counts
                    extension_counts[extension] += 1
                    folder_counts[relative_path][extension] += 1
                    
                    # Add to file list
                    file_list.append({
                        'path': relative_path,
                        'name': file,
                        'extension': extension,
                        'size_kb': round(os.path.getsize(file_path) / 1024, 2)
                    })
                except Exception as e:
                    print(f"Warning: Could not process {file}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error walking directory {base_path}: {e}")
        return None, None, None
        
    return dict(extension_counts), dict(folder_counts), file_list

def main():
    # Directory to analyze
    base_path = r"C:\alstom"
    
    # Check if path exists
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist!")
        sys.exit(1)
    
    print(f"Analyzing directory: {base_path}")
    print("-" * 50)
    
    # Get file counts
    ext_counts, folder_counts, files = count_files(base_path)
    
    if not ext_counts:
        print("No files found or error occurred!")
        sys.exit(1)
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare report data
    report = {
        'timestamp': datetime.now().isoformat(),
        'base_path': base_path,
        'total_files': sum(ext_counts.values()),
        'extensions': ext_counts,
        'folders': folder_counts,
        'files': files
    }
    
    # Save report
    report_path = os.path.join(output_dir, 'file_analysis.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\nSummary:")
    print(f"Total files: {report['total_files']}")
    print("\nFiles by extension:")
    for ext, count in sorted(ext_counts.items()):
        print(f"{ext or '(no extension)':<15} {count:>5}")
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
