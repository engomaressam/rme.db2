# Source and destination base directories
$sourceBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\07.ml.models"
$destBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db.data\07.ml.models"

# File extensions to move (ML models, data files, and other large files)
$extensions = @(
    # Model files
    "*.pkl", "*.joblib", "*.h5", "*.hdf5", "*.pb", "*.pt", "*.pth", "*.model", "*.tflite",
    # Data files
    "*.csv", "*.parquet", "*.feather", "*.hdf", "*.npy", "*.npz", "*.tfrecords",
    # Archives
    "*.7z", "*.zip", "*.rar", "*.tar", "*.gz",
    # Other data files
    "*.db", "*.sqlite", "*.sqlite3", "*.data", "*.dat"
)

# Get all files matching the extensions recursively, excluding .ipynb files
$filesToMove = Get-ChildItem -Path $sourceBase -Recurse -Include $extensions | 
    Where-Object { $_.Extension -ne '.ipynb' }

# Add large text files (>1MB) to the files to move
$largeTextFiles = Get-ChildItem -Path $sourceBase -Recurse -Include "*.txt" | Where-Object { $_.Length -gt 1MB }
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

Write-Host "All ML model and data files have been moved successfully!"
Write-Host "Moved $($filesToMove.Count) files in total"
Write-Host "Source: $sourceBase"
Write-Host "Destination: $destBase"
