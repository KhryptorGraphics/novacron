# Setup Go Environment
Write-Host "Setting up Go environment..." -ForegroundColor Green

# Set Go binary path
$goBinPath = "C:\Users\kp\scoop\apps\go\current\bin"

# Add to current session PATH
$env:PATH = $env:PATH + ";$goBinPath"

# Verify Go installation
Write-Host "Go version:" -ForegroundColor Yellow
& "$goBinPath\go.exe" version

Write-Host "Go environment:" -ForegroundColor Yellow
& "$goBinPath\go.exe" env GOPATH
& "$goBinPath\go.exe" env GOROOT

# Create alias for easier use
Set-Alias -Name go -Value "$goBinPath\go.exe"
Set-Alias -Name gofmt -Value "$goBinPath\gofmt.exe"

Write-Host "Go is now ready to use in this session!" -ForegroundColor Green
Write-Host "You can now use 'go version' or other go commands." -ForegroundColor Cyan

# Test the setup
Write-Host "`nTesting Go setup:" -ForegroundColor Yellow
go version