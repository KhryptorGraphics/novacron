#!/bin/bash

# NovaCron Cache Infrastructure Setup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    print_status "Waiting for $service_name at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z $host $port 2>/dev/null; then
            print_success "$service_name is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within $((max_attempts * 2)) seconds"
    return 1
}

# Function to setup Redis Sentinel configuration
setup_sentinel() {
    local mode=$1
    
    print_status "Setting up Redis with Sentinel..."
    
    # Start Redis Sentinel setup
    docker-compose -f docker-compose.cache.yml up -d redis-master redis-slave-1 redis-slave-2
    
    # Wait for Redis instances
    wait_for_service localhost 6379 "Redis Master"
    wait_for_service localhost 6380 "Redis Slave 1"
    wait_for_service localhost 6381 "Redis Slave 2"
    
    # Start Sentinels
    docker-compose -f docker-compose.cache.yml up -d redis-sentinel-1 redis-sentinel-2 redis-sentinel-3
    
    # Wait for Sentinels
    wait_for_service localhost 26379 "Redis Sentinel 1"
    wait_for_service localhost 26380 "Redis Sentinel 2"
    wait_for_service localhost 26381 "Redis Sentinel 3"
    
    print_success "Redis Sentinel cluster is ready!"
}

# Function to setup Redis Cluster configuration
setup_cluster() {
    print_status "Setting up Redis Cluster..."
    
    # Start Redis cluster nodes
    docker-compose -f docker-compose.cache.yml --profile cluster up -d
    
    # Wait for cluster nodes
    for port in 7001 7002 7003 7004 7005 7006; do
        wait_for_service localhost $port "Redis Cluster Node $port"
    done
    
    # Initialize cluster
    print_status "Initializing Redis Cluster..."
    docker-compose -f docker-compose.cache.yml --profile cluster up redis-cluster-init
    
    print_success "Redis Cluster is ready!"
}

# Function to setup standalone Redis
setup_standalone() {
    print_status "Setting up standalone Redis..."
    
    # Start just Redis master
    docker-compose -f docker-compose.cache.yml up -d redis-master
    
    # Wait for Redis
    wait_for_service localhost 6379 "Redis Master"
    
    print_success "Standalone Redis is ready!"
}

# Function to start monitoring services
start_monitoring() {
    print_status "Starting cache monitoring services..."
    
    # Start Redis Commander
    docker-compose -f docker-compose.cache.yml up -d redis-commander
    wait_for_service localhost 8081 "Redis Commander"
    
    # Start cache monitor
    docker-compose -f docker-compose.cache.yml up -d cache-monitor
    wait_for_service localhost 9091 "Cache Monitor"
    
    print_success "Monitoring services are ready!"
    print_status "Redis Commander available at: http://localhost:8081"
    print_status "Cache Monitor available at: http://localhost:8082"
    print_status "Metrics endpoint available at: http://localhost:9091/metrics"
}

# Function to run cache tests
run_tests() {
    print_status "Running cache connectivity tests..."
    
    # Test Redis connectivity
    if command_exists redis-cli; then
        print_status "Testing Redis Master..."
        redis-cli -p 6379 ping
        
        if docker-compose -f docker-compose.cache.yml ps | grep -q redis-slave; then
            print_status "Testing Redis Slaves..."
            redis-cli -p 6380 ping
            redis-cli -p 6381 ping
        fi
        
        if docker-compose -f docker-compose.cache.yml ps | grep -q redis-sentinel; then
            print_status "Testing Redis Sentinels..."
            redis-cli -p 26379 sentinel masters
        fi
    else
        print_warning "redis-cli not found, skipping direct Redis tests"
    fi
    
    # Test cache monitor health endpoint
    print_status "Testing cache monitor health endpoint..."
    if command_exists curl; then
        curl -f http://localhost:9091/health || print_warning "Cache monitor health check failed"
        curl -f http://localhost:9091/metrics || print_warning "Cache monitor metrics endpoint failed"
    else
        print_warning "curl not found, skipping HTTP tests"
    fi
    
    print_success "Cache tests completed!"
}

# Function to show status
show_status() {
    print_status "Cache Infrastructure Status:"
    echo
    
    # Show running containers
    if command_exists docker-compose; then
        docker-compose -f docker-compose.cache.yml ps
        echo
    fi
    
    # Show service endpoints
    print_status "Service Endpoints:"
    echo "  Redis Master:      redis://localhost:6379"
    echo "  Redis Slave 1:     redis://localhost:6380" 
    echo "  Redis Slave 2:     redis://localhost:6381"
    echo "  Redis Sentinel 1:  redis://localhost:26379"
    echo "  Redis Sentinel 2:  redis://localhost:26380"
    echo "  Redis Sentinel 3:  redis://localhost:26381"
    echo "  Redis Commander:   http://localhost:8081"
    echo "  Cache Monitor:     http://localhost:8082"
    echo "  Metrics Endpoint:  http://localhost:9091/metrics"
    echo "  Health Endpoint:   http://localhost:9091/health"
    echo
}

# Function to cleanup
cleanup() {
    print_status "Cleaning up cache infrastructure..."
    
    docker-compose -f docker-compose.cache.yml down -v
    docker-compose -f docker-compose.cache.yml --profile cluster down -v
    
    # Remove volumes if requested
    if [ "$1" = "--volumes" ]; then
        print_status "Removing persistent volumes..."
        docker volume prune -f
    fi
    
    print_success "Cleanup completed!"
}

# Function to show help
show_help() {
    echo "NovaCron Cache Infrastructure Setup Script"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  setup [MODE]     Setup cache infrastructure"
    echo "    sentinel       Redis with Sentinel (High Availability)"
    echo "    cluster        Redis Cluster (Horizontal Scaling)"
    echo "    standalone     Single Redis instance"
    echo "  start            Start monitoring services"
    echo "  test             Run connectivity tests"
    echo "  status           Show infrastructure status"
    echo "  cleanup          Stop and remove containers"
    echo "  help             Show this help message"
    echo
    echo "Options:"
    echo "  --volumes        Remove volumes during cleanup"
    echo
    echo "Examples:"
    echo "  $0 setup sentinel    # Setup Redis with Sentinel HA"
    echo "  $0 setup cluster     # Setup Redis Cluster"
    echo "  $0 start             # Start monitoring services"
    echo "  $0 test              # Test cache connectivity"
    echo "  $0 cleanup --volumes # Remove everything including data"
}

# Main script logic
main() {
    local command=${1:-help}
    local option=${2:-}
    
    case $command in
        setup)
            case $option in
                sentinel)
                    setup_sentinel
                    ;;
                cluster)
                    setup_cluster
                    ;;
                standalone)
                    setup_standalone
                    ;;
                *)
                    print_error "Invalid setup mode. Use: sentinel, cluster, or standalone"
                    exit 1
                    ;;
            esac
            ;;
        start)
            start_monitoring
            ;;
        test)
            run_tests
            ;;
        status)
            show_status
            ;;
        cleanup)
            cleanup $option
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    if ! command_exists docker; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        print_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    if ! command_exists nc; then
        print_warning "netcat (nc) is recommended for service health checks"
    fi
}

# Script entry point
if [ "$0" = "${BASH_SOURCE[0]}" ]; then
    check_prerequisites
    main "$@"
fi