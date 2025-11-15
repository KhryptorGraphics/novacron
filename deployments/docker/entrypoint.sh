#!/bin/bash
set -e

echo "🚀 Starting Novacron Onboarding System..."

# Run database migrations if enabled
if [ "${RUN_MIGRATIONS}" = "true" ]; then
    echo "📊 Running database migrations..."
    migrate -path /app/migrations \
            -database "${DATABASE_URL}" \
            up
    echo "✅ Migrations completed"
fi

# Wait for dependencies
if [ -n "${WAIT_FOR_POSTGRES}" ]; then
    echo "⏳ Waiting for PostgreSQL..."
    until pg_isready -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}"; do
        echo "Waiting for database connection..."
        sleep 2
    done
    echo "✅ PostgreSQL is ready"
fi

if [ -n "${WAIT_FOR_REDIS}" ]; then
    echo "⏳ Waiting for Redis..."
    until redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping; then
        echo "Waiting for Redis connection..."
        sleep 2
    done
    echo "✅ Redis is ready"
fi

# Execute the main command
echo "🎯 Starting application server..."
exec "$@"
