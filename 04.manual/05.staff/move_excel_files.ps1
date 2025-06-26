# Source and destination base directories
$sourceBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\04.manual\05.staff"
$destBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db.data\04.manual\05.staff"

# Get all Excel, CSV, PDF, archive, executable, and image files recursively
$extensions = @("*.xlsx", "*.xls", "*.xlsb", "*.csv", "*.pdf", "*.7z", "*.zip", "*.rar", "*.exe", 
                "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.tif", "*.webp", "*.svg")
$filesToMove = Get-ChildItem -Path $sourceBase -Recurse -Include $extensions

# Add large text files (>1MB) to the files to move
$largeTextFiles = Get-ChildItem -Path $sourceBase -Recurse -Include "*.txt" | Where-Object { $_.Length -gt 1MB }
$filesToMove = @($filesToMove) + @($largeTextFiles)

# Remove any duplicate files that might be included in both lists
$filesToMove = $filesToMove | Sort-Object FullName -Unique

foreach ($file in $filesToMove) {
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
    Move-Item -Path $file.FullName -Destination $destPath -Force
}

Write-Host "All Excel, PDF, archive, and executable files have been moved successfully!"
