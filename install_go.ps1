# Go Installation Script for Windows
# This script will download and install the latest version of Go

Write-Host "Installing Go Programming Language..." -ForegroundColor Green

# Define Go version (update this to the latest version)
$goVersion = "1.21.5"
$goUrl = "https://golang.org/dl/go$goVersion.windows-amd64.msi"
$tempPath = "$env:TEMP\go$goVersion.windows-amd64.msi"

try {
    # Download Go installer
    Write-Host "Downloading Go $goVersion..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $goUrl -OutFile $tempPath -UseBasicParsing
    
    # Install Go
    Write-Host "Installing Go..." -ForegroundColor Yellow
    Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $tempPath, "/quiet" -Wait
    
    # Clean up
    Remove-Item $tempPath -Force
    
    # Add Go to PATH if not already there
    $goPath = "C:\Program Files\Go\bin"
    $currentPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    
    if ($currentPath -notlike "*$goPath*") {
        Write-Host "Adding Go to system PATH..." -ForegroundColor Yellow
        $newPath = $currentPath + ";" + $goPath
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "Machine")
    }
    
    # Set GOPATH
    $goWorkspace = "$env:USERPROFILE\go"
    [Environment]::SetEnvironmentVariable("GOPATH", $goWorkspace, "User")
    
    Write-Host "Go installation completed successfully!" -ForegroundColor Green
    Write-Host "Please restart your PowerShell session or run: refreshenv" -ForegroundColor Cyan
    Write-Host "Then verify installation with: go version" -ForegroundColor Cyan
    
} catch {
    Write-Host "Error installing Go: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please try manual installation from https://golang.org/dl/" -ForegroundColor Yellow
}