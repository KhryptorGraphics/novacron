# NovaCron Dependencies Setup Script
# This script installs Go and Node.js if they're not available

param(
    [switch]$Force
)

Write-Host "🔧 NovaCron Dependencies Setup" -ForegroundColor Blue
Write-Host "==============================" -ForegroundColor Blue

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Install-WithWinget {
    param([string]$Package, [string]$Name)
    
    if (Test-Command "winget") {
        Write-Host "Installing $Name using winget..." -ForegroundColor Yellow
        winget install $Package --accept-package-agreements --accept-source-agreements
        return $true
    }
    return $false
}

function Install-WithChocolatey {
    param([string]$Package, [string]$Name)
    
    if (Test-Command "choco") {
        Write-Host "Installing $Name using Chocolatey..." -ForegroundColor Yellow
        choco install $Package -y
        return $true
    }
    return $false
}

function Install-Chocolatey {
    Write-Host "Installing Chocolatey package manager..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
}

# Check current status
Write-Host "`n📋 Checking current installation status..." -ForegroundColor Cyan

$goInstalled = Test-Command "go"
$nodeInstalled = Test-Command "node"
$npmInstalled = Test-Command "npm"

Write-Host "Go: $(if($goInstalled) { '✅ Installed' } else { '❌ Not found' })" -ForegroundColor $(if($goInstalled) { 'Green' } else { 'Red' })
Write-Host "Node.js: $(if($nodeInstalled) { '✅ Installed' } else { '❌ Not found' })" -ForegroundColor $(if($nodeInstalled) { 'Green' } else { 'Red' })
Write-Host "npm: $(if($npmInstalled) { '✅ Installed' } else { '❌ Not found' })" -ForegroundColor $(if($npmInstalled) { 'Green' } else { 'Red' })

if ($goInstalled -and $nodeInstalled -and $npmInstalled -and -not $Force) {
    Write-Host "`n🎉 All dependencies are already installed!" -ForegroundColor Green
    exit 0
}

Write-Host "`n🚀 Installing missing dependencies..." -ForegroundColor Yellow

# Install Go if missing
if (-not $goInstalled -or $Force) {
    Write-Host "`n📦 Installing Go..." -ForegroundColor Cyan
    
    $installed = $false
    
    # Try winget first
    if (Install-WithWinget "GoLang.Go" "Go") {
        $installed = $true
    }
    # Try chocolatey
    elseif (Install-WithChocolatey "golang" "Go") {
        $installed = $true
    }
    # Install chocolatey and try again
    elseif (-not (Test-Command "choco")) {
        Install-Chocolatey
        if (Install-WithChocolatey "golang" "Go") {
            $installed = $true
        }
    }
    
    if (-not $installed) {
        Write-Host "❌ Failed to install Go automatically. Please install manually from https://golang.org/dl/" -ForegroundColor Red
        Write-Host "   1. Download Go 1.21+ for Windows" -ForegroundColor Yellow
        Write-Host "   2. Run the installer" -ForegroundColor Yellow
        Write-Host "   3. Restart PowerShell" -ForegroundColor Yellow
    } else {
        Write-Host "✅ Go installation completed" -ForegroundColor Green
    }
}

# Install Node.js if missing
if (-not $nodeInstalled -or $Force) {
    Write-Host "`n📦 Installing Node.js..." -ForegroundColor Cyan
    
    $installed = $false
    
    # Try winget first
    if (Install-WithWinget "OpenJS.NodeJS" "Node.js") {
        $installed = $true
    }
    # Try chocolatey
    elseif (Install-WithChocolatey "nodejs" "Node.js") {
        $installed = $true
    }
    
    if (-not $installed) {
        Write-Host "❌ Failed to install Node.js automatically. Please install manually from https://nodejs.org/" -ForegroundColor Red
        Write-Host "   1. Download Node.js 18+ LTS for Windows" -ForegroundColor Yellow
        Write-Host "   2. Run the installer" -ForegroundColor Yellow
        Write-Host "   3. Restart PowerShell" -ForegroundColor Yellow
    } else {
        Write-Host "✅ Node.js installation completed" -ForegroundColor Green
    }
}

Write-Host "`n🔄 Refreshing environment..." -ForegroundColor Cyan

# Refresh environment variables
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")

Write-Host "`n📋 Verifying installations..." -ForegroundColor Cyan

# Test installations
$goWorking = Test-Command "go"
$nodeWorking = Test-Command "node"
$npmWorking = Test-Command "npm"

Write-Host "Go: $(if($goWorking) { '✅ Working' } else { '❌ Not working' })" -ForegroundColor $(if($goWorking) { 'Green' } else { 'Red' })
Write-Host "Node.js: $(if($nodeWorking) { '✅ Working' } else { '❌ Not working' })" -ForegroundColor $(if($nodeWorking) { 'Green' } else { 'Red' })
Write-Host "npm: $(if($npmWorking) { '✅ Working' } else { '❌ Not working' })" -ForegroundColor $(if($npmWorking) { 'Green' } else { 'Red' })

if ($goWorking -and $nodeWorking -and $npmWorking) {
    Write-Host "`n🎉 All dependencies are now installed and working!" -ForegroundColor Green
    Write-Host "`n🚀 You can now run: .\start_development.ps1" -ForegroundColor Cyan
} else {
    Write-Host "`n⚠️  Some dependencies may require a system restart to work properly." -ForegroundColor Yellow
    Write-Host "   Please restart your computer and try again." -ForegroundColor Yellow
}

Write-Host "`n📚 Next Steps:" -ForegroundColor Blue
Write-Host "   1. Restart PowerShell (or your computer if needed)" -ForegroundColor White
Write-Host "   2. Run: .\start_development.ps1" -ForegroundColor White
Write-Host "   3. Access the dashboard at http://localhost:8092" -ForegroundColor White