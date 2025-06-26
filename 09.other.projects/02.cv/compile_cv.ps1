$ErrorActionPreference = "Stop"
$currentLocation = Get-Location
Write-Host "Current directory: $currentLocation"

Write-Host "Changing to CV directory..."
Set-Location -Path $PSScriptRoot
Write-Host "New directory: $(Get-Location)"

Write-Host "Compiling LaTeX file..."
Write-Host "Running: pdflatex -interaction=nonstopmode `"latex of cv.rev.05.tex`""
& pdflatex -interaction=nonstopmode "latex of cv.rev.05.tex"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation successful!"
    $pdfPath = Join-Path (Get-Location) "latex of cv.rev.05.pdf"
    if (Test-Path $pdfPath) {
        Write-Host "PDF file created at: $pdfPath"
    } else {
        Write-Host "Warning: PDF file not found at expected location: $pdfPath"
    }
} else {
    Write-Host "Compilation failed with error code $LASTEXITCODE"
    Write-Host "Check the log file for more details"
}

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 