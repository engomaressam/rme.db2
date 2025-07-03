# Similar Videos Detector Runner

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Using $pythonVersion"
} catch {
    Write-Host "Python is not installed or not in PATH. Please install Python first."
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
    exit
}

# Display header
Write-Host "Similar Videos Detector" -ForegroundColor Cyan
Write-Host "=====================" -ForegroundColor Cyan
Write-Host ""

# Install dependencies if needed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt
Write-Host "Dependencies installed." -ForegroundColor Green
Write-Host ""

# We'll use the GUI folder browser built into the Python script
Write-Host "A folder selection dialog will appear. Please select your videos folder." -ForegroundColor Yellow

# Similarity threshold
Write-Host ""
Write-Host "Enter similarity threshold (0.0-1.0, recommended: 0.85):"
Write-Host "Higher values mean stricter matching. Press Enter for default (0.85):"
$threshold = Read-Host

if ([string]::IsNullOrWhiteSpace($threshold)) {
    $threshold = 0.85
}

# Duration tolerance
Write-Host ""
Write-Host "Enter duration tolerance in seconds (recommended: 1.0):"
Write-Host "This is how many seconds difference is allowed when grouping by duration. Press Enter for default (1.0):"
$duration = Read-Host

if ([string]::IsNullOrWhiteSpace($duration)) {
    $duration = 1.0
}

# Run the detector
Write-Host ""
Write-Host "Running detector with the following parameters:" -ForegroundColor Cyan
Write-Host "  Directory: $scanDir"
Write-Host "  Similarity threshold: $threshold"
Write-Host "  Duration tolerance: $duration"
Write-Host ""

# Execute the Python script with GUI mode
python similar_videos_detector.py --gui --threshold $threshold --duration-tolerance $duration

# Keep console open
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
