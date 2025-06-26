# Source and destination base directories
$sourceBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\09.other.projects"
$destBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db.data\09.other.projects"

# File extensions to move (common large file types in projects)
$extensions = @(
    # Documents
    "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx",
    # Media files
    "*.mp4", "*.avi", "*.mov", "*.wmv", "*.flv", "*.mkv",
    "*.mp3", "*.wav", "*.wma", "*.aac", "*.flac",
    "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp", "*.svg", "*.psd", "*.ai",
    # Archives
    "*.7z", "*.zip", "*.rar", "*.tar", "*.gz",
    # Data files
    "*.csv", "*.json", "*.xml", "*.db", "*.sqlite", "*.sqlite3", "*.data", "*.dat",
    # Executables and installers
    "*.exe", "*.msi", "*.dmg", "*.pkg",
    # Virtual environments and dependencies
    "*.whl", "*.egg", "*.pyc", "__pycache__", "node_modules"
)

# Get all files matching the extensions recursively, excluding .ipynb files
$filesToMove = Get-ChildItem -Path $sourceBase -Recurse -Include $extensions -ErrorAction SilentlyContinue | 
    Where-Object { $_.Extension -ne '.ipynb' }

# Add large text and log files (>1MB) to the files to move
$largeTextFiles = Get-ChildItem -Path $sourceBase -Recurse -Include @("*.txt", "*.log") -ErrorAction SilentlyContinue | 
    Where-Object { $_.Length -gt 1MB }
$filesToMove = @($filesToMove) + @($largeTextFiles)

# Remove any duplicate files that might be included in both lists
$filesToMove = $filesToMove | Sort-Object FullName -Unique

if ($filesToMove.Count -eq 0) {
    Write-Host "No files to move in $sourceBase."
    exit
}

Write-Host "Found $($filesToMove.Count) files to move from $sourceBase to $destBase"

foreach ($file in $filesToMove) {
    try {
        # Skip directories (like __pycache__ and node_modules)
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

# Handle special directories (like __pycache__ and node_modules) - delete them after moving their contents
$specialDirs = @("__pycache__", "node_modules")
foreach ($dir in $specialDirs) {
    $dirsToRemove = Get-ChildItem -Path $sourceBase -Directory -Recurse -Filter $dir -ErrorAction SilentlyContinue
    foreach ($dirToRemove in $dirsToRemove) {
        try {
            if (Test-Path $dirToRemove.FullName) {
                Write-Host "Removing directory: $($dirToRemove.FullName)"
                Remove-Item -Path $dirToRemove.FullName -Recurse -Force -ErrorAction Stop
            }
        }
        catch {
            Write-Host "Error removing directory $($dirToRemove.FullName): $_" -ForegroundColor Red
        }
    }
}

Write-Host "All large files and directories have been moved/cleaned up successfully!"
Write-Host "Moved $($filesToMove.Count) files in total"
Write-Host "Source: $sourceBase"
Write-Host "Destination: $destBase"
