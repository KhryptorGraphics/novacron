#!/bin/bash
# Runbook CLI - Command-line interface for runbook automation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNBOOK_ENGINE="${SCRIPT_DIR}/runbook-engine.py"
RUNBOOK_DIR="/tmp/runbooks"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Runbook Automation CLI

Usage: $(basename "$0") <command> [options]

Commands:
  generate <type> <context>   Generate a new runbook
  execute <runbook_id>        Execute a runbook
  validate <runbook_id>       Validate a runbook
  list [tag]                  List runbooks (optionally filter by tag)
  show <runbook_id>           Show runbook details
  test <runbook_id>           Test runbook execution (dry-run)
  help                        Show this help message

Examples:
  # Generate incident runbook
  $(basename "$0") generate incident '{"type":"pod_crash","pod_name":"api-123"}'

  # Execute a runbook
  $(basename "$0") execute rb-incident-1234567890

  # List all runbooks
  $(basename "$0") list

  # Test runbook without execution
  $(basename "$0") test rb-incident-1234567890

EOF
}

# Generate runbook
generate_runbook() {
    local type=$1
    local context=$2

    log_info "Generating ${type} runbook..."

    python3 -c "
import sys
import json
sys.path.append('${SCRIPT_DIR}')
from runbook_engine import RunbookGenerator, RunbookLibrary
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RunbookGenerator')

generator = RunbookGenerator(logger)
library = RunbookLibrary('${RUNBOOK_DIR}', logger)

context_data = json.loads('${context}')

if '${type}' == 'incident':
    runbook = generator.generate_from_incident(context_data)
else:
    runbook = generator.generate_from_pattern('${type}', context_data)

filepath = library.save(runbook)
print(f'Runbook generated: {runbook.id}')
print(f'Saved to: {filepath}')
"
}

# Execute runbook
execute_runbook() {
    local runbook_id=$1
    local dry_run=${2:-false}

    log_info "Executing runbook: ${runbook_id}"

    python3 -c "
import sys
sys.path.append('${SCRIPT_DIR}')
from runbook_engine import RunbookExecutor, RunbookLibrary
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RunbookExecutor')

library = RunbookLibrary('${RUNBOOK_DIR}', logger)
executor = RunbookExecutor(logger, dry_run=${dry_run})

try:
    runbook = library.load('${runbook_id}')
    success = executor.execute(runbook)
    sys.exit(0 if success else 1)
except Exception as e:
    logger.error(f'Execution failed: {str(e)}')
    sys.exit(1)
"
}

# Validate runbook
validate_runbook() {
    local runbook_id=$1

    log_info "Validating runbook: ${runbook_id}"

    python3 -c "
import sys
sys.path.append('${SCRIPT_DIR}')
from runbook_engine import RunbookValidator, RunbookLibrary
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('RunbookValidator')

library = RunbookLibrary('${RUNBOOK_DIR}', logger)
validator = RunbookValidator(logger)

try:
    runbook = library.load('${runbook_id}')
    is_valid, errors = validator.validate(runbook)

    if is_valid:
        print('✓ Runbook is valid')
        sys.exit(0)
    else:
        print('✗ Runbook validation failed:')
        for error in errors:
            print(f'  - {error}')
        sys.exit(1)
except Exception as e:
    logger.error(f'Validation failed: {str(e)}')
    sys.exit(1)
"
}

# List runbooks
list_runbooks() {
    local tag=${1:-}

    log_info "Listing runbooks..."

    python3 -c "
import sys
sys.path.append('${SCRIPT_DIR}')
from runbook_engine import RunbookLibrary
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('RunbookLibrary')

library = RunbookLibrary('${RUNBOOK_DIR}', logger)

tag = '${tag}' if '${tag}' else None
runbooks = library.list(tag=tag)

if runbooks:
    print(f'Found {len(runbooks)} runbooks:')
    for rb_id in runbooks:
        print(f'  - {rb_id}')
else:
    print('No runbooks found')
"
}

# Show runbook details
show_runbook() {
    local runbook_id=$1

    python3 -c "
import sys
import yaml
sys.path.append('${SCRIPT_DIR}')
from runbook_engine import RunbookLibrary
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('RunbookLibrary')

library = RunbookLibrary('${RUNBOOK_DIR}', logger)

try:
    runbook = library.load('${runbook_id}')

    print(f'Runbook: {runbook.name}')
    print(f'ID: {runbook.id}')
    print(f'Type: {runbook.type.value}')
    print(f'Version: {runbook.version}')
    print(f'Description: {runbook.description}')
    print(f'Steps: {len(runbook.steps)}')
    print()
    print('Steps:')
    for i, step in enumerate(runbook.steps, 1):
        print(f'  {i}. {step.name}')
        print(f'     Command: {step.command}')
        if step.dependencies:
            print(f'     Dependencies: {step.dependencies}')

except Exception as e:
    print(f'Error: {str(e)}')
    sys.exit(1)
"
}

# Main command dispatcher
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    case "$1" in
        generate)
            if [ $# -lt 3 ]; then
                log_error "Usage: $0 generate <type> <context_json>"
                exit 1
            fi
            generate_runbook "$2" "$3"
            ;;
        execute)
            if [ $# -lt 2 ]; then
                log_error "Usage: $0 execute <runbook_id>"
                exit 1
            fi
            execute_runbook "$2" "False"
            ;;
        test)
            if [ $# -lt 2 ]; then
                log_error "Usage: $0 test <runbook_id>"
                exit 1
            fi
            log_info "Running in DRY-RUN mode (no actual execution)"
            execute_runbook "$2" "True"
            ;;
        validate)
            if [ $# -lt 2 ]; then
                log_error "Usage: $0 validate <runbook_id>"
                exit 1
            fi
            validate_runbook "$2"
            ;;
        list)
            list_runbooks "${2:-}"
            ;;
        show)
            if [ $# -lt 2 ]; then
                log_error "Usage: $0 show <runbook_id>"
                exit 1
            fi
            show_runbook "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"
