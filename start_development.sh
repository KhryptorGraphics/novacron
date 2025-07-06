#!/bin/bash

# NovaCron Development Environment Startup Script
# This script starts both the backend API server and frontend development server

set -e

echo "🚀 Starting NovaCron Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Go
    if ! command -v go &> /dev/null; then
        print_error "Go is not installed. Please install Go 1.19 or later."
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18 or later."
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm."
        exit 1
    fi
    
    print_success "All dependencies are available"
}

# Setup Go modules
setup_backend() {
    print_status "Setting up backend..."
    
    # Initialize Go module if not exists
    if [ ! -f "go.mod" ]; then
        print_status "Initializing Go module..."
        go mod init github.com/khryptorgraphics/novacron
    fi
    
    # Install required Go dependencies
    print_status "Installing Go dependencies..."
    go mod tidy
    
    # Install additional dependencies that might be missing
    go get github.com/gorilla/mux@latest
    go get github.com/gorilla/websocket@latest
    go get github.com/gorilla/handlers@latest
    go get github.com/digitalocean/go-libvirt@latest
    go get github.com/google/uuid@latest
    
    print_success "Backend setup complete"
}

# Setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Install npm dependencies
    if [ ! -d "node_modules" ]; then
        print_status "Installing npm dependencies..."
        npm install
    else
        print_status "Updating npm dependencies..."
        npm update
    fi
    
    cd ..
    print_success "Frontend setup complete"
}

# Start backend server
start_backend() {
    print_status "Starting backend API server..."
    
    # Build and run the API server
    cd backend/cmd/api-server
    go build -o ../../../novacron-api .
    cd ../../..
    
    # Start the API server in background
    ./novacron-api &
    BACKEND_PID=$!
    
    # Wait a moment for server to start
    sleep 2
    
    # Check if backend is running
    if kill -0 $BACKEND_PID 2>/dev/null; then
        print_success "Backend API server started (PID: $BACKEND_PID)"
        echo $BACKEND_PID > .backend.pid
    else
        print_error "Failed to start backend API server"
        exit 1
    fi
}

# Start frontend development server
start_frontend() {
    print_status "Starting frontend development server..."
    
    cd frontend
    
    # Start the frontend development server in background
    npm run dev &
    FRONTEND_PID=$!
    
    cd ..
    
    # Wait a moment for server to start
    sleep 3
    
    # Check if frontend is running
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        print_success "Frontend development server started (PID: $FRONTEND_PID)"
        echo $FRONTEND_PID > .frontend.pid
    else
        print_error "Failed to start frontend development server"
        exit 1
    fi
}

# Health check function
health_check() {
    print_status "Performing health checks..."
    
    # Check backend health
    if curl -s http://localhost:8080/health > /dev/null; then
        print_success "Backend API is healthy"
    else
        print_warning "Backend API health check failed"
    fi
    
    # Check if frontend is accessible (this might take a moment)
    sleep 5
    if curl -s http://localhost:3000 > /dev/null; then
        print_success "Frontend is accessible"
    else
        print_warning "Frontend accessibility check failed (may still be starting)"
    fi
}

# Cleanup function
cleanup() {
    print_status "Cleaning up processes..."
    
    if [ -f .backend.pid ]; then
        BACKEND_PID=$(cat .backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            print_status "Stopped backend server (PID: $BACKEND_PID)"
        fi
        rm .backend.pid
    fi
    
    if [ -f .frontend.pid ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            print_status "Stopped frontend server (PID: $FRONTEND_PID)"
        fi
        rm .frontend.pid
    fi
    
    # Clean up any remaining processes
    pkill -f "novacron-api" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    
    print_success "Cleanup complete"
}

# Trap to cleanup on script exit
trap cleanup EXIT

# Main execution
main() {
    print_status "NovaCron Development Environment Setup"
    print_status "======================================"
    
    check_dependencies
    setup_backend
    setup_frontend
    start_backend
    start_frontend
    health_check
    
    print_success "🎉 Development environment is ready!"
    echo ""
    echo "📊 Frontend Dashboard: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:8080"
    echo "📋 API Documentation: http://localhost:8080/api/info"
    echo "💚 Health Check: http://localhost:8080/health"
    echo ""
    print_status "Press Ctrl+C to stop all services"
    
    # Keep script running
    while true; do
        sleep 1
    done
}

# Handle command line arguments
case "${1:-}" in
    "stop")
        print_status "Stopping development environment..."
        cleanup
        exit 0
        ;;
    "status")
        print_status "Checking service status..."
        if [ -f .backend.pid ] && kill -0 $(cat .backend.pid) 2>/dev/null; then
            print_success "Backend is running (PID: $(cat .backend.pid))"
        else
            print_warning "Backend is not running"
        fi
        
        if [ -f .frontend.pid ] && kill -0 $(cat .frontend.pid) 2>/dev/null; then
            print_success "Frontend is running (PID: $(cat .frontend.pid))"
        else
            print_warning "Frontend is not running"
        fi
        exit 0
        ;;
    "help"|"-h"|"--help")
        echo "NovaCron Development Environment"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  (no args)  Start the development environment"
        echo "  stop       Stop all running services"
        echo "  status     Check status of services"
        echo "  help       Show this help message"
        exit 0
        ;;
    *)
        main
        ;;
esac