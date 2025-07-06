# NovaCron Development Environment Startup Script (PowerShell)
# This script starts both the backend API server and frontend development server

param(
    [string]$Command = "start"
)

# Colors for output
$ErrorActionPreference = "Stop"

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if required tools are installed
function Test-Dependencies {
    Write-Status "Checking dependencies..."
    
    # Check Go
    try {
        $goVersion = go version
        Write-Success "Go is installed: $goVersion"
    } catch {
        Write-Error "Go is not installed. Please install Go 1.19 or later."
        exit 1
    }
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        Write-Success "Node.js is installed: $nodeVersion"
    } catch {
        Write-Error "Node.js is not installed. Please install Node.js 18 or later."
        exit 1
    }
    
    # Check npm
    try {
        $npmVersion = npm --version
        Write-Success "npm is installed: $npmVersion"
    } catch {
        Write-Error "npm is not installed. Please install npm."
        exit 1
    }
}

# Setup Go modules
function Initialize-Backend {
    Write-Status "Setting up backend..."
    
    # Initialize Go module if not exists
    if (-not (Test-Path "go.mod")) {
        Write-Status "Initializing Go module..."
        go mod init github.com/khryptorgraphics/novacron
    }
    
    # Install required Go dependencies
    Write-Status "Installing Go dependencies..."
    go mod tidy
    
    # Install additional dependencies
    go get github.com/gorilla/mux@latest
    go get github.com/gorilla/websocket@latest
    go get github.com/gorilla/handlers@latest
    go get github.com/digitalocean/go-libvirt@latest
    go get github.com/google/uuid@latest
    
    Write-Success "Backend setup complete"
}

# Setup frontend
function Initialize-Frontend {
    Write-Status "Setting up frontend..."
    
    Push-Location frontend
    
    try {
        # Install npm dependencies
        if (-not (Test-Path "node_modules")) {
            Write-Status "Installing npm dependencies..."
            npm install
        } else {
            Write-Status "Updating npm dependencies..."
            npm update
        }
        
        Write-Success "Frontend setup complete"
    } finally {
        Pop-Location
    }
}

# Start backend server
function Start-Backend {
    Write-Status "Starting backend API server..."
    
    # Build the API server
    Push-Location "backend/cmd/api-server"
    try {
        go build -o "../../../novacron-api.exe" .
        Write-Success "Backend built successfully"
    } finally {
        Pop-Location
    }
    
    # Start the API server
    $backendProcess = Start-Process -FilePath "./novacron-api.exe" -PassThru -WindowStyle Hidden
    
    if ($backendProcess) {
        Write-Success "Backend API server started (PID: $($backendProcess.Id))"
        $backendProcess.Id | Out-File -FilePath ".backend.pid" -Encoding ASCII
        
        # Wait for server to start
        Start-Sleep -Seconds 3
        
        # Test if server is responding
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Success "Backend API is responding"
            }
        } catch {
            Write-Warning "Backend API may still be starting up"
        }
    } else {
        Write-Error "Failed to start backend API server"
        exit 1
    }
}

# Start frontend development server
function Start-Frontend {
    Write-Status "Starting frontend development server..."
    
    Push-Location frontend
    try {
        # Start the frontend development server
        $frontendProcess = Start-Process -FilePath "npm" -ArgumentList "run", "dev" -PassThru -WindowStyle Hidden
        
        if ($frontendProcess) {
            Write-Success "Frontend development server started (PID: $($frontendProcess.Id))"
            $frontendProcess.Id | Out-File -FilePath "../.frontend.pid" -Encoding ASCII
            
            # Wait for server to start
            Start-Sleep -Seconds 5
            
            Write-Success "Frontend should be available at http://localhost:3000"
        } else {
            Write-Error "Failed to start frontend development server"
            exit 1
        }
    } finally {
        Pop-Location
    }
}

# Health check function
function Test-Services {
    Write-Status "Performing health checks..."
    
    # Check backend health
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Backend API is healthy"
        }
    } catch {
        Write-Warning "Backend API health check failed: $($_.Exception.Message)"
    }
    
    # Check frontend (may take longer to start)
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Frontend is accessible"
        }
    } catch {
        Write-Warning "Frontend accessibility check failed (may still be starting): $($_.Exception.Message)"
    }
}

# Cleanup function
function Stop-Services {
    Write-Status "Stopping services..."
    
    # Stop backend
    if (Test-Path ".backend.pid") {
        $backendPid = Get-Content ".backend.pid"
        try {
            Stop-Process -Id $backendPid -Force
            Write-Status "Stopped backend server (PID: $backendPid)"
        } catch {
            Write-Warning "Could not stop backend process: $($_.Exception.Message)"
        }
        Remove-Item ".backend.pid" -Force
    }
    
    # Stop frontend
    if (Test-Path ".frontend.pid") {
        $frontendPid = Get-Content ".frontend.pid"
        try {
            Stop-Process -Id $frontendPid -Force
            Write-Status "Stopped frontend server (PID: $frontendPid)"
        } catch {
            Write-Warning "Could not stop frontend process: $($_.Exception.Message)"
        }
        Remove-Item ".frontend.pid" -Force
    }
    
    # Clean up any remaining processes
    Get-Process | Where-Object { $_.ProcessName -like "*novacron-api*" } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Write-Success "Cleanup complete"
}

# Check service status
function Get-ServiceStatus {
    Write-Status "Checking service status..."
    
    if (Test-Path ".backend.pid") {
        $backendPid = Get-Content ".backend.pid"
        try {
            $process = Get-Process -Id $backendPid -ErrorAction Stop
            Write-Success "Backend is running (PID: $backendPid)"
        } catch {
            Write-Warning "Backend is not running"
        }
    } else {
        Write-Warning "Backend is not running"
    }
    
    if (Test-Path ".frontend.pid") {
        $frontendPid = Get-Content ".frontend.pid"
        try {
            $process = Get-Process -Id $frontendPid -ErrorAction Stop
            Write-Success "Frontend is running (PID: $frontendPid)"
        } catch {
            Write-Warning "Frontend is not running"
        }
    } else {
        Write-Warning "Frontend is not running"
    }
}

# Main execution
function Start-Development {
    Write-Status "NovaCron Development Environment Setup"
    Write-Status "======================================"
    
    Test-Dependencies
    Initialize-Backend
    Initialize-Frontend
    Start-Backend
    Start-Frontend
    Test-Services
    
    Write-Success "🎉 Development environment is ready!"
    Write-Host ""
    Write-Host "📊 Frontend Dashboard: http://localhost:3000" -ForegroundColor Cyan
    Write-Host "🔧 Backend API: http://localhost:8080" -ForegroundColor Cyan
    Write-Host "📋 API Documentation: http://localhost:8080/api/info" -ForegroundColor Cyan
    Write-Host "💚 Health Check: http://localhost:8080/health" -ForegroundColor Cyan
    Write-Host ""
    Write-Status "Press Ctrl+C to stop all services"
    
    # Keep script running
    try {
        while ($true) {
            Start-Sleep -Seconds 1
        }
    } finally {
        Stop-Services
    }
}

# Handle command line arguments
switch ($Command.ToLower()) {
    "stop" {
        Write-Status "Stopping development environment..."
        Stop-Services
        break
    }
    "status" {
        Get-ServiceStatus
        break
    }
    "help" {
        Write-Host "NovaCron Development Environment"
        Write-Host ""
        Write-Host "Usage: .\start_development.ps1 [command]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  start      Start the development environment (default)"
        Write-Host "  stop       Stop all running services"
        Write-Host "  status     Check status of services"
        Write-Host "  help       Show this help message"
        break
    }
    default {
        Start-Development
        break
    }
}