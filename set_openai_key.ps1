# PowerShell script to set the OpenAI API key for Qdrant integration

# Check if API key is provided as argument
param (
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ApiKey
)

# Check if .env file exists, create it if not
if (-not (Test-Path .env)) {
    New-Item -Path .env -ItemType File | Out-Null
    Write-Host "Created new .env file"
}

# Read the current content of .env
$envContent = Get-Content .env -ErrorAction SilentlyContinue

# Check if OPENAI_API_KEY is already set in .env
$keyExists = $false
$newContent = @()

foreach ($line in $envContent) {
    if ($line -match "^OPENAI_API_KEY=") {
        $newContent += "OPENAI_API_KEY=$ApiKey"
        $keyExists = $true
        Write-Host "Updated OPENAI_API_KEY in .env file"
    } else {
        $newContent += $line
    }
}

# Add key to .env if it doesn't exist
if (-not $keyExists) {
    $newContent += "OPENAI_API_KEY=$ApiKey"
    Write-Host "Added OPENAI_API_KEY to .env file"
}

# Write back to .env file
$newContent | Set-Content .env

# Set for current PowerShell session
$env:OPENAI_API_KEY = $ApiKey
Write-Host "Set OPENAI_API_KEY for current PowerShell session"

Write-Host "OpenAI API key has been configured successfully!"
Write-Host "To set in a new PowerShell session, run: `$env:OPENAI_API_KEY = (Get-Content .env | Select-String 'OPENAI_API_KEY=').ToString().Split('=')[1]"

# Display instructions for loading in Go applications
Write-Host "`nTo verify and test the Qdrant OpenAI integration:"
Write-Host "1. Run 'cd tools/indexer'"
Write-Host "2. Run 'go run test_qdrant.go' to verify Qdrant connection"
Write-Host "3. Run 'go run main.go' to index files with OpenAI embeddings"
