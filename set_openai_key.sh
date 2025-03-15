#!/bin/bash
# Script to set the OpenAI API key for Qdrant integration

# Check if API key is provided as argument
if [ $# -lt 1 ]; then
    echo "Usage: ./set_openai_key.sh YOUR_OPENAI_API_KEY"
    echo "Example: ./set_openai_key.sh sk-abcdefg123456789"
    exit 1
fi

API_KEY=$1

# Check if .env file exists, create it if not
if [ ! -f .env ]; then
    touch .env
    echo "Created new .env file"
fi

# Check if OPENAI_API_KEY is already set in .env
if grep -q "OPENAI_API_KEY" .env; then
    # Update existing key
    sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$API_KEY/" .env
    echo "Updated OPENAI_API_KEY in .env file"
else
    # Add key to .env
    echo "OPENAI_API_KEY=$API_KEY" >> .env
    echo "Added OPENAI_API_KEY to .env file"
fi

# Set for current shell session
export OPENAI_API_KEY=$API_KEY
echo "Set OPENAI_API_KEY for current shell session"

echo "OpenAI API key has been configured successfully!"
echo "To use in new shell sessions, run: source .env"
