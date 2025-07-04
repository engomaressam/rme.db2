# Source and destination base directories
$sourceBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\18.LLMs"
$destBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db.data\18.LLMs"

# File extensions to move (common large files in LLM projects)
$extensions = @(
    # Model files
    "*.bin", "*.pth", "*.pt", "*.h5", "*.hdf5", "*.safetensors", "*.gguf", "*.ggml",
    # Vector databases and embeddings
    "*.faiss", "*.pkl", "*.joblib", "*.npy", "*.npz", "*.parquet",
    # Data files
    "*.jsonl", "*.jsonl.gz", "*.csv", "*.xlsx", "*.xlsb", "*.parquet", "*.arrow",
    # Archives
    "*.zip", "*.7z", "*.rar", "*.tar", "*.gz",
    # Virtual environments and caches
    "venv*", "env*", ".venv*", "__pycache__", ".pytest_cache", ".mypy_cache",
    # Logs and checkpoints
    "*.log", "*.ckpt", "checkpoint*", "runs/*", "logs/*", "wandb/*", "mlruns/*",
    # Images
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff", "*.webp"
)

# Get all files matching the extensions recursively, excluding .git directories
$filesToMove = Get-ChildItem -Path $sourceBase -Recurse -Include $extensions -ErrorAction SilentlyContinue | 
    Where-Object { 
        $_.Extension -ne '.ipynb' -and 
        $_.FullName -notmatch '\\.git\\' -and
        $_.FullName -notmatch 'node_modules\\.bin'  # Don't move .bin files in node_modules
    }

# Add large text and log files (>1MB) to the files to move
$largeTextFiles = Get-ChildItem -Path $sourceBase -Recurse -Include @("*.txt", "*.log") -ErrorAction SilentlyContinue | 
    Where-Object { $_.Length -gt 1MB }
$filesToMove = @($filesToMove) + @($largeTextFiles)

# Remove any duplicate files that might be included in both lists
$filesToMove = $filesToMove | Sort-Object FullName -Unique

# Find directories to remove after moving their contents
$dirsToRemove = @("venv*", "env*", ".venv*", "__pycache__", ".pytest_cache", ".mypy_cache",
                 "wandb", "mlruns", "runs", "logs", "checkpoints", "models")

# Get all directories that match our patterns
$directoriesToProcess = $dirsToRemove | ForEach-Object {
    Get-ChildItem -Path $sourceBase -Directory -Recurse -Filter $_ -ErrorAction SilentlyContinue
} | Sort-Object FullName -Unique

if ($filesToMove.Count -eq 0 -and $directoriesToProcess.Count -eq 0) {
    Write-Host "No files or directories to process in $sourceBase."
    exit
}

Write-Host "Found $($filesToMove.Count) files to move from $sourceBase to $destBase"

# Move files
foreach ($file in $filesToMove) {
    try {
        # Skip directories (handled separately)
        if ($file.PSIsContainer) {
            continue
        }
        
        # Get the relative path from the source base
        $relativePath = $file.FullName.Substring($sourceBase.Length + 1)
        
        # Create the destination directory if it doesn't exist
        $destDir = Join-Path $destBase (Split-Path $relativePath -Parent)
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        # Move the file
        $destPath = Join-Path $destBase $relativePath
        Write-Host "Moving $($file.FullName) to $destPath"
        Move-Item -Path $file.FullName -Destination $destPath -Force -ErrorAction Stop
    }
    catch {
        Write-Host "Error moving $($file.FullName): $_" -ForegroundColor Red
    }
}

# Process directories (move their contents and then remove the directories)
foreach ($dir in $directoriesToProcess) {
    try {
        # Skip if directory doesn't exist or is empty
        if (-not (Test-Path $dir.FullName) -or (Get-ChildItem -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
            continue
        }
        
        # Get the relative path from the source base
        $relativePath = $dir.FullName.Substring($sourceBase.Length + 1)
        
        # Create the destination directory
        $destDir = Join-Path $destBase $relativePath
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        # Move all items from the source directory to the destination
        Write-Host "Moving contents of $($dir.FullName) to $destDir"
        Get-ChildItem -Path $dir.FullName -Force | Move-Item -Destination $destDir -Force -ErrorAction Stop
        
        # Remove the now-empty directory
        Write-Host "Removing empty directory: $($dir.FullName)"
        Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction Stop
    }
    catch {
        Write-Host "Error processing directory $($dir.FullName): $_" -ForegroundColor Red
    }
}

Write-Host "All LLM-related files and directories have been moved/cleaned up successfully!"
Write-Host "Moved $($filesToMove.Count) files in total"
Write-Host "Source: $sourceBase"
Write-Host "Destination: $destBase"
