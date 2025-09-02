#\!/bin/bash

# NovaCron Production Validation Tests
echo '=== NovaCron Production Environment Validation ==='
echo 'Starting comprehensive validation tests...'
echo

# Test 1: Service Health Checks
echo '1. SERVICE HEALTH CHECKS'
echo '------------------------'
services=('postgres:15555' 'redis:15560' 'prometheus:15564' 'grafana:15565')
for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    if nc -z localhost $port 2>/dev/null; then
        echo "✅ $name: HEALTHY (port $port)"
    else
        echo "❌ $name: UNHEALTHY (port $port)"
    fi
done
echo

# Test 2: Database Operations
echo '2. DATABASE CONNECTIVITY TEST'
echo '-----------------------------'
docker exec novacron-postgres-1 psql -U postgres -d novacron -c "SELECT 'Database connection successful' as status;" 2>/dev/null && echo '✅ PostgreSQL: Connection and query successful' || echo '❌ PostgreSQL: Connection failed'
echo

# Test 3: Redis Operations
echo '3. REDIS CONNECTIVITY TEST'
echo '--------------------------'
docker exec novacron-redis-1 redis-cli -a redis123 ping 2>/dev/null | grep -q PONG && echo '✅ Redis: Connection and ping successful' || echo '❌ Redis: Connection failed'
echo

# Test 4: Prometheus Metrics
echo '4. PROMETHEUS METRICS TEST'
echo '--------------------------'
curl -s http://localhost:15564/api/v1/query?query=up | grep -q '"status":"success"' && echo '✅ Prometheus: Metrics API responding' || echo '❌ Prometheus: Metrics API failed'
echo

# Test 5: Grafana API
echo '5. GRAFANA API TEST'
echo '------------------'
curl -s http://localhost:15565/api/health | grep -q 'ok' && echo '✅ Grafana: Health API responding' || echo '❌ Grafana: Health API failed'
echo

echo '=== VALIDATION COMPLETE ==='

