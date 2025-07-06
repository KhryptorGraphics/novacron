# Install Go and test the NovaCron server
Write-Host "🚀 Setting up NovaCron development environment..." -ForegroundColor Green

# Check if Go is already installed
try {
    $goVersion = & go version 2>$null
    if ($goVersion) {
        Write-Host "✅ Go is already installed: $goVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "📦 Installing Go..." -ForegroundColor Yellow
    
    # Download Go installer for Windows
    $goUrl = "https://go.dev/dl/go1.21.5.windows-amd64.msi"
    $tempPath = "$env:TEMP\go-installer.msi"
    
    try {
        Write-Host "⬇️  Downloading Go installer..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $goUrl -OutFile $tempPath -UseBasicParsing
        
        Write-Host "🔧 Installing Go (this may take a moment)..." -ForegroundColor Yellow
        Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $tempPath, "/quiet", "/norestart" -Wait
        
        # Add Go to PATH for current session
        $goPath = "C:\Program Files\Go\bin"
        $env:PATH = "$env:PATH;$goPath"
        
        # Clean up installer
        Remove-Item $tempPath -Force -ErrorAction SilentlyContinue
        
        Write-Host "✅ Go installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to install Go: $_" -ForegroundColor Red
        Write-Host "Please install Go manually from https://golang.org/dl/" -ForegroundColor Yellow
        exit 1
    }
}

# Test Go installation
try {
    Write-Host "🧪 Testing Go installation..." -ForegroundColor Yellow
    $goVersion = & go version
    Write-Host "✅ Go version: $goVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Go installation test failed. Please restart your terminal and try again." -ForegroundColor Red
    exit 1
}

# Initialize Go modules
Write-Host "📦 Initializing Go modules..." -ForegroundColor Yellow
try {
    & go mod tidy
    Write-Host "✅ Go modules initialized" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Go mod tidy failed, but continuing..." -ForegroundColor Yellow
}

Write-Host "🎉 Environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start the server: go run cmd/novacron/main.go" -ForegroundColor White
Write-Host "2. Test endpoints: go run tmp_rovodev_test_server.go" -ForegroundColor White
Write-Host "3. Start frontend: cd frontend && npm run dev" -ForegroundColor White