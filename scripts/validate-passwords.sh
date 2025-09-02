#!/bin/bash

# Validate password strength in .env.production

ENV_FILE=".env.production"

check_password_strength() {
    local password=$1
    local name=$2
    local min_length=12
    
    # Check length
    if [ ${#password} -lt $min_length ]; then
        echo "❌ $name is too short (minimum $min_length characters)"
        return 1
    fi
    
    # Check for uppercase
    if ! [[ "$password" =~ [A-Z] ]]; then
        echo "❌ $name missing uppercase letters"
        return 1
    fi
    
    # Check for lowercase
    if ! [[ "$password" =~ [a-z] ]]; then
        echo "❌ $name missing lowercase letters"
        return 1
    fi
    
    # Check for numbers
    if ! [[ "$password" =~ [0-9] ]]; then
        echo "❌ $name missing numbers"
        return 1
    fi
    
    # Check for special characters
    if ! [[ "$password" =~ [^a-zA-Z0-9] ]]; then
        echo "❌ $name missing special characters"
        return 1
    fi
    
    echo "✓ $name meets security requirements"
    return 0
}

# Extract and validate passwords from .env.production
if [ -f "$ENV_FILE" ]; then
    echo "Validating password strength..."
    
    DB_PASSWORD=$(grep "^DB_PASSWORD=" "$ENV_FILE" | cut -d'=' -f2)
    REDIS_PASSWORD=$(grep "^REDIS_PASSWORD=" "$ENV_FILE" | cut -d'=' -f2)
    ADMIN_PASSWORD=$(grep "^ADMIN_PASSWORD=" "$ENV_FILE" | cut -d'=' -f2)
    
    check_password_strength "$DB_PASSWORD" "Database password"
    check_password_strength "$REDIS_PASSWORD" "Redis password"
    check_password_strength "$ADMIN_PASSWORD" "Admin password"
else
    echo "Error: $ENV_FILE not found"
    exit 1
fi
