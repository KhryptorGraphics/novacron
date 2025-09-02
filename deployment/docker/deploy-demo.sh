#!/bin/bash

# NovaCron Demo Deployment Script
set -e

echo "🚀 Starting NovaCron Demo Deployment..."

# Check Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Clean up any existing containers
echo "🧹 Cleaning up existing containers..."
docker-compose -f docker-compose.demo.yml down --remove-orphans --volumes 2>/dev/null || true

# Remove old images to ensure fresh build
echo "🗑️ Removing old images..."
docker rmi -f $(docker images | grep 'novacron-mock' | awk '{print $3}') 2>/dev/null || true

# Build and start services
echo "🏗️ Building and starting services..."
docker-compose -f docker-compose.demo.yml up --build -d

echo "⏳ Waiting for services to become healthy..."

# Function to check service health
check_service_health() {
    local service_name=$1
    local port=$2
    local path=${3:-"/health"}
    local max_attempts=30
    local attempt=1
    
    echo "🔍 Checking $service_name on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:$port$path" > /dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts - waiting for $service_name..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to become healthy"
    return 1
}

# Check database
echo "🔍 Checking PostgreSQL database..."
for i in {1..12}; do
    if docker-compose -f docker-compose.demo.yml exec -T postgres pg_isready -U novacron > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    echo "⏳ Waiting for PostgreSQL... ($i/12)"
    sleep 5
done

# Check Redis
echo "🔍 Checking Redis cache..."
for i in {1..6}; do
    if docker-compose -f docker-compose.demo.yml exec -T redis redis-cli ping | grep -q PONG; then
        echo "✅ Redis is ready"
        break
    fi
    echo "⏳ Waiting for Redis... ($i/6)"
    sleep 5
done

# Check API service
check_service_health "Mock API" 15561

# Check Frontend service
check_service_health "Frontend" 15566 "/"

# Check Prometheus
check_service_health "Prometheus" 15564 "/-/healthy"

# Check Grafana
check_service_health "Grafana" 15565 "/api/health"

echo ""
echo "🎉 NovaCron Demo Deployment Complete!"
echo ""
echo "📊 Service URLs:"
echo "   • Frontend Dashboard: http://localhost:15566"
echo "   • API Health Check:   http://localhost:15561/health"
echo "   • PostgreSQL:         localhost:15555 (novacron/novacron123)"
echo "   • Redis:              localhost:15560"
echo "   • Prometheus:         http://localhost:15564"
echo "   • Grafana:            http://localhost:15565 (admin/admin123)"
echo ""
echo "🔐 Demo Accounts:"
echo "   • admin / admin (Admin access)"
echo "   • operator1 / password (Operator access)"
echo "   • user1 / password (User access)"
echo ""
echo "🔄 To stop all services:"
echo "   docker-compose -f docker-compose.demo.yml down"
echo ""
echo "📋 To view logs:"
echo "   docker-compose -f docker-compose.demo.yml logs -f"
echo ""

# Show service status
echo "📈 Service Status:"
docker-compose -f docker-compose.demo.yml ps

echo ""
echo "✨ Ready for testing!"