#!/bin/bash
# DWCP v3 Configuration Drift Detection and Remediation
# Continuously monitors configuration and automatically remediates drift

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${PROJECT_ROOT}/config"
DEPLOYMENTS_DIR="${PROJECT_ROOT}/deployments"
STATE_DIR="/var/lib/dwcp/drift-detection"
LOG_FILE="/var/log/dwcp/drift-detection.log"
ALERT_EMAIL="${ALERT_EMAIL:-ops@novacron.io}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"

# Detection settings
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"  # 5 minutes
AUTO_REMEDIATE="${AUTO_REMEDIATE:-true}"
DRY_RUN="${DRY_RUN:-false}"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date -Iseconds)
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Initialize
init() {
    log_info "Initializing drift detection system"

    # Create directories
    mkdir -p "$STATE_DIR" /var/log/dwcp

    # Store baseline configuration
    if [[ ! -f "$STATE_DIR/baseline.json" ]]; then
        log_info "Creating baseline configuration snapshot"
        capture_baseline
    fi

    log_success "Drift detection initialized"
}

# Capture baseline configuration
capture_baseline() {
    log_info "Capturing baseline configuration"

    local baseline="$STATE_DIR/baseline.json"

    cat > "$baseline" <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "config_files": {},
    "terraform_state": {},
    "running_config": {}
}
EOF

    # Capture configuration files
    log_info "Capturing configuration file checksums"
    local config_checksums=$(find "$CONFIG_DIR" -type f -name "*.yaml" -exec sha256sum {} \; | \
        jq -R -s -c 'split("\n") | map(select(length > 0) | split("  ")) | map({(.[1]): .[0]}) | add')

    jq --argjson checksums "$config_checksums" '.config_files = $checksums' "$baseline" > "$baseline.tmp" && \
        mv "$baseline.tmp" "$baseline"

    # Capture Terraform state (if available)
    if command -v terraform &> /dev/null && [[ -d "$DEPLOYMENTS_DIR/terraform/environments/production" ]]; then
        log_info "Capturing Terraform state"
        cd "$DEPLOYMENTS_DIR/terraform/environments/production"
        terraform show -json > "$STATE_DIR/terraform-state.json" 2>/dev/null || true
    fi

    # Capture running configuration from live nodes
    log_info "Capturing running configuration from nodes"
    capture_running_config

    log_success "Baseline captured successfully"
}

# Capture running configuration from nodes
capture_running_config() {
    local running_config="$STATE_DIR/running-config.json"

    # Initialize running config
    echo '{}' > "$running_config"

    # Get list of DWCP nodes from inventory
    if [[ -f "$DEPLOYMENTS_DIR/ansible/inventory/production.ini" ]]; then
        local nodes=$(ansible all -i "$DEPLOYMENTS_DIR/ansible/inventory/production.ini" --list-hosts 2>/dev/null | \
            grep -v "hosts (" | tr -d ' ' || echo "")

        if [[ -n "$nodes" ]]; then
            log_info "Found nodes: $(echo "$nodes" | tr '\n' ' ')"

            # Collect configuration from each node
            for node in $nodes; do
                log_info "Collecting config from $node"

                # Get DWCP configuration
                local node_config=$(ansible "$node" -i "$DEPLOYMENTS_DIR/ansible/inventory/production.ini" \
                    -m shell -a "cat /etc/dwcp/*.yaml 2>/dev/null || echo '{}'" 2>/dev/null | \
                    grep -A 1000 "SUCCESS" | tail -n +2 || echo "{}")

                # Get service status
                local service_status=$(ansible "$node" -i "$DEPLOYMENTS_DIR/ansible/inventory/production.ini" \
                    -m shell -a "systemctl is-active dwcp" 2>/dev/null | \
                    grep -A 1 "SUCCESS" | tail -n 1 || echo "unknown")

                # Store in running config
                jq --arg node "$node" --arg config "$node_config" --arg status "$service_status" \
                    '.[$node] = {"config": $config, "status": $status}' "$running_config" > "$running_config.tmp" && \
                    mv "$running_config.tmp" "$running_config"
            done
        fi
    fi
}

# Detect drift
detect_drift() {
    log_info "Starting drift detection"

    local drift_detected=false
    local drift_report="$STATE_DIR/drift-report-$(date +%Y%m%d-%H%M%S).json"

    # Initialize report
    cat > "$drift_report" <<EOF
{
    "timestamp": "$(date -Iseconds)",
    "drifts": []
}
EOF

    # Check configuration file drift
    log_info "Checking configuration file drift"
    check_config_drift "$drift_report" && drift_detected=true

    # Check Terraform drift
    log_info "Checking Terraform state drift"
    check_terraform_drift "$drift_report" && drift_detected=true

    # Check running configuration drift
    log_info "Checking running configuration drift"
    check_running_config_drift "$drift_report" && drift_detected=true

    # Check policy compliance
    log_info "Checking policy compliance"
    check_policy_compliance "$drift_report" && drift_detected=true

    if [[ "$drift_detected" == true ]]; then
        log_warn "Configuration drift detected"

        # Generate drift summary
        local drift_count=$(jq '.drifts | length' "$drift_report")
        log_warn "Total drifts detected: $drift_count"

        # Send alert
        send_alert "$drift_report"

        # Auto-remediate if enabled
        if [[ "$AUTO_REMEDIATE" == "true" ]]; then
            log_info "Auto-remediation enabled, attempting to fix drift"
            remediate_drift "$drift_report"
        else
            log_warn "Auto-remediation disabled, manual intervention required"
        fi

        return 1
    else
        log_success "No configuration drift detected"
        return 0
    fi
}

# Check configuration file drift
check_config_drift() {
    local drift_report=$1
    local drift=false

    local baseline="$STATE_DIR/baseline.json"
    local current_checksums=$(find "$CONFIG_DIR" -type f -name "*.yaml" -exec sha256sum {} \; | \
        jq -R -s -c 'split("\n") | map(select(length > 0) | split("  ")) | map({(.[1]): .[0]}) | add')

    # Compare with baseline
    local baseline_checksums=$(jq -r '.config_files' "$baseline")

    # Find changed files
    local changed_files=$(jq -n --argjson baseline "$baseline_checksums" --argjson current "$current_checksums" \
        '$baseline | to_entries | map(select(.value != $current[.key])) | map(.key)')

    if [[ $(echo "$changed_files" | jq 'length') -gt 0 ]]; then
        log_warn "Configuration files have been modified"
        drift=true

        # Add to drift report
        jq --argjson files "$changed_files" \
            '.drifts += [{
                "type": "config_file",
                "severity": "medium",
                "description": "Configuration files modified",
                "files": $files
            }]' "$drift_report" > "$drift_report.tmp" && mv "$drift_report.tmp" "$drift_report"
    fi

    $drift && return 0 || return 1
}

# Check Terraform drift
check_terraform_drift() {
    local drift_report=$1
    local drift=false

    if ! command -v terraform &> /dev/null; then
        log_info "Terraform not available, skipping Terraform drift check"
        return 1
    fi

    cd "$DEPLOYMENTS_DIR/terraform/environments/production"

    # Run terraform plan to detect drift
    log_info "Running terraform plan to detect infrastructure drift"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run mode, skipping actual Terraform check"
        return 1
    fi

    terraform plan -detailed-exitcode -out=/tmp/tfplan 2>&1 | tee /tmp/tfplan.log || {
        local exit_code=$?
        if [[ $exit_code -eq 2 ]]; then
            log_warn "Terraform drift detected (exit code 2)"
            drift=true

            # Extract changes
            local changes=$(grep -A 5 "Terraform will perform" /tmp/tfplan.log || echo "Changes detected")

            # Add to drift report
            jq --arg changes "$changes" \
                '.drifts += [{
                    "type": "terraform",
                    "severity": "high",
                    "description": "Infrastructure drift detected",
                    "details": $changes
                }]' "$drift_report" > "$drift_report.tmp" && mv "$drift_report.tmp" "$drift_report"
        fi
    }

    $drift && return 0 || return 1
}

# Check running configuration drift
check_running_config_drift() {
    local drift_report=$1
    local drift=false

    # Capture current running config
    local current_running="$STATE_DIR/running-config-current.json"
    capture_running_config
    mv "$STATE_DIR/running-config.json" "$current_running"

    # Compare with baseline
    local baseline_running="$STATE_DIR/baseline.json"

    if [[ -f "$baseline_running" ]]; then
        local diff=$(diff <(jq -S . "$baseline_running") <(jq -S . "$current_running") || true)

        if [[ -n "$diff" ]]; then
            log_warn "Running configuration has drifted from baseline"
            drift=true

            # Add to drift report
            jq --arg diff "$diff" \
                '.drifts += [{
                    "type": "running_config",
                    "severity": "high",
                    "description": "Running configuration differs from baseline",
                    "diff": $diff
                }]' "$drift_report" > "$drift_report.tmp" && mv "$drift_report.tmp" "$drift_report"
        fi
    fi

    $drift && return 0 || return 1
}

# Check policy compliance
check_policy_compliance() {
    local drift_report=$1
    local drift=false

    if ! command -v opa &> /dev/null; then
        log_info "OPA not available, skipping policy compliance check"
        return 1
    fi

    log_info "Checking OPA policy compliance"

    # Check each configuration file against policies
    for config_file in "$CONFIG_DIR"/*.yaml; do
        local mode=$(basename "$config_file" | sed 's/dwcp-v3-//' | sed 's/.yaml//')

        log_info "Checking $mode mode configuration against policies"

        # Convert YAML to JSON for OPA
        local config_json=$(python3 -c "import yaml, json, sys; print(json.dumps(yaml.safe_load(open('$config_file'))))" 2>/dev/null || echo "{}")

        # Check security policy
        local security_result=$(opa eval --data "$PROJECT_ROOT/policies/dwcp-v3-security.rego" \
            --input <(echo "$config_json") --format pretty "data.dwcp.security.decision" 2>/dev/null || echo '{"allow": false}')

        if [[ $(echo "$security_result" | jq -r '.allow') != "true" ]]; then
            log_warn "Policy violation detected in $config_file"
            drift=true

            local violations=$(echo "$security_result" | jq -r '.violations[]' || echo "Unknown violation")

            # Add to drift report
            jq --arg file "$config_file" --arg violations "$violations" \
                '.drifts += [{
                    "type": "policy_violation",
                    "severity": "critical",
                    "description": "Security policy violation",
                    "file": $file,
                    "violations": $violations
                }]' "$drift_report" > "$drift_report.tmp" && mv "$drift_report.tmp" "$drift_report"
        fi

        # Check network policy
        local network_result=$(opa eval --data "$PROJECT_ROOT/policies/dwcp-v3-network.rego" \
            --input <(echo "$config_json") --format pretty "data.dwcp.network.decision" 2>/dev/null || echo '{"allow": false}')

        if [[ $(echo "$network_result" | jq -r '.allow') != "true" ]]; then
            log_warn "Network policy violation detected in $config_file"
            drift=true

            local violations=$(echo "$network_result" | jq -r '.violations[]' || echo "Unknown violation")

            # Add to drift report
            jq --arg file "$config_file" --arg violations "$violations" \
                '.drifts += [{
                    "type": "policy_violation",
                    "severity": "critical",
                    "description": "Network policy violation",
                    "file": $file,
                    "violations": $violations
                }]' "$drift_report" > "$drift_report.tmp" && mv "$drift_report.tmp" "$drift_report"
        fi
    done

    $drift && return 0 || return 1
}

# Remediate drift
remediate_drift() {
    local drift_report=$1

    log_info "Starting drift remediation"

    local drifts=$(jq -r '.drifts[] | @json' "$drift_report")

    while IFS= read -r drift; do
        local type=$(echo "$drift" | jq -r '.type')
        local severity=$(echo "$drift" | jq -r '.severity')

        log_info "Remediating $type drift (severity: $severity)"

        case $type in
            config_file)
                remediate_config_drift "$drift"
                ;;
            terraform)
                remediate_terraform_drift "$drift"
                ;;
            running_config)
                remediate_running_config_drift "$drift"
                ;;
            policy_violation)
                remediate_policy_violation "$drift"
                ;;
            *)
                log_warn "Unknown drift type: $type"
                ;;
        esac
    done <<< "$drifts"

    log_success "Drift remediation completed"
}

# Remediate configuration file drift
remediate_config_drift() {
    local drift=$1

    log_info "Remediating configuration file drift"

    # Restore from Git (assuming configuration is version controlled)
    if [[ -d "$PROJECT_ROOT/.git" ]]; then
        log_info "Restoring configuration files from Git"

        if [[ "$DRY_RUN" == "false" ]]; then
            cd "$PROJECT_ROOT"
            git checkout HEAD -- config/
            log_success "Configuration files restored from Git"
        else
            log_info "Dry run: would restore configuration files from Git"
        fi
    else
        log_warn "Git repository not found, cannot auto-remediate configuration drift"
    fi
}

# Remediate Terraform drift
remediate_terraform_drift() {
    local drift=$1

    log_info "Remediating Terraform drift"

    if [[ "$DRY_RUN" == "false" ]]; then
        cd "$DEPLOYMENTS_DIR/terraform/environments/production"
        terraform apply -auto-approve /tmp/tfplan
        log_success "Terraform drift remediated"
    else
        log_info "Dry run: would apply Terraform plan"
    fi
}

# Remediate running configuration drift
remediate_running_config_drift() {
    local drift=$1

    log_info "Remediating running configuration drift"

    # Re-apply Ansible playbook
    if [[ "$DRY_RUN" == "false" ]]; then
        ansible-playbook -i "$DEPLOYMENTS_DIR/ansible/inventory/production.ini" \
            "$DEPLOYMENTS_DIR/ansible/dwcp-v3-setup.yml" \
            --tags config
        log_success "Running configuration remediated via Ansible"
    else
        log_info "Dry run: would run Ansible playbook"
    fi
}

# Remediate policy violation
remediate_policy_violation() {
    local drift=$1

    log_error "Policy violation detected - manual intervention required"
    log_error "Details: $(echo "$drift" | jq -r '.violations')"

    # Policy violations require manual review
    send_critical_alert "Policy violation requires manual review" "$drift"
}

# Send alert
send_alert() {
    local drift_report=$1

    log_info "Sending drift alert"

    local drift_count=$(jq '.drifts | length' "$drift_report")
    local summary=$(jq -r '.drifts[] | "\(.type) (\(.severity))"' "$drift_report" | head -5)

    # Email alert
    if command -v mail &> /dev/null; then
        cat <<EOF | mail -s "[DWCP] Configuration Drift Detected ($drift_count drifts)" "$ALERT_EMAIL"
Configuration drift detected in DWCP v3 deployment.

Drift Count: $drift_count
Timestamp: $(date -Iseconds)

Summary:
$summary

Full report: $drift_report

Remediation: $(if [[ "$AUTO_REMEDIATE" == "true" ]]; then echo "Automatic"; else echo "Manual Required"; fi)
EOF
    fi

    # Slack alert (if webhook configured)
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        curl -X POST "$SLACK_WEBHOOK" -H 'Content-Type: application/json' -d @- <<EOF
{
    "text": ":warning: Configuration Drift Detected",
    "blocks": [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Configuration Drift Alert*\n\nDrift Count: $drift_count\nTimestamp: $(date -Iseconds)"
            }
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Summary:\n\`\`\`$summary\`\`\`"
            }
        }
    ]
}
EOF
    fi
}

# Send critical alert
send_critical_alert() {
    local title=$1
    local details=$2

    log_error "Sending critical alert: $title"

    # Email
    if command -v mail &> /dev/null; then
        echo "$details" | mail -s "[DWCP CRITICAL] $title" "$ALERT_EMAIL"
    fi

    # Slack
    if [[ -n "$SLACK_WEBHOOK" ]]; then
        curl -X POST "$SLACK_WEBHOOK" -H 'Content-Type: application/json' -d @- <<EOF
{
    "text": ":rotating_light: CRITICAL ALERT: $title",
    "attachments": [
        {
            "color": "danger",
            "text": "$details"
        }
    ]
}
EOF
    fi
}

# Main monitoring loop
monitor() {
    log_info "Starting drift detection monitor (interval: ${CHECK_INTERVAL}s)"

    while true; do
        detect_drift || {
            log_warn "Drift detection cycle completed with findings"
        }

        log_info "Sleeping for ${CHECK_INTERVAL} seconds"
        sleep "$CHECK_INTERVAL"
    done
}

# CLI
usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
    init            Initialize drift detection system
    baseline        Capture new baseline configuration
    detect          Run drift detection once
    monitor         Continuous drift monitoring
    remediate       Manually trigger remediation

Options:
    --dry-run       Perform detection without remediation
    --no-auto       Disable automatic remediation
    --interval N    Set check interval (seconds, default: 300)

Examples:
    $0 init
    $0 detect --dry-run
    $0 monitor --interval 600
    $0 remediate

EOF
}

# Parse arguments
COMMAND=${1:-}
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-auto)
            AUTO_REMEDIATE=false
            shift
            ;;
        --interval)
            CHECK_INTERVAL=$2
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute command
case $COMMAND in
    init)
        init
        ;;
    baseline)
        capture_baseline
        ;;
    detect)
        init
        detect_drift
        ;;
    monitor)
        init
        monitor
        ;;
    remediate)
        init
        if [[ -f "$STATE_DIR/drift-report-"*.json ]]; then
            latest_report=$(ls -t "$STATE_DIR"/drift-report-*.json | head -1)
            remediate_drift "$latest_report"
        else
            log_error "No drift report found. Run 'detect' first."
            exit 1
        fi
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
