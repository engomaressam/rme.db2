# Source and destination base directories
$dataBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db.data\07.ml.models"
$repoBase = "C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\07.ml.models"

# Find all .ipynb files in the data directory
$ipynbFiles = Get-ChildItem -Path $dataBase -Recurse -Filter "*.ipynb"

if ($ipynbFiles.Count -eq 0) {
    Write-Host "No .ipynb files found in the data directory."
    exit
}

Write-Host "Found $($ipynbFiles.Count) .ipynb files to move back to the repository."

foreach ($file in $ipynbFiles) {
    try {
        # Get the relative path from the data base
        $relativePath = $file.FullName.Substring($dataBase.Length + 1)
        
        # Construct the destination path in the repo
        $destPath = Join-Path $repoBase $relativePath
        $destDir = Split-Path -Parent $destPath
        
        # Create the destination directory if it doesn't exist
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        # Move the file back to the repo
        Write-Host "Moving $($file.FullName) back to $destPath"
        Move-Item -Path $file.FullName -Destination $destPath -Force -ErrorAction Stop
    }
    catch {
        Write-Host "Error moving $($file.FullName): $_" -ForegroundColor Red
    }
}

Write-Host "All .ipynb files have been moved back to the repository successfully!"
Write-Host "Moved $($ipynbFiles.Count) files in total"
