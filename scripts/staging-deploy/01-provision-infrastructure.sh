#!/usr/bin/env bash
# DWCP v3 Phase 5: Staging Infrastructure Provisioning
# Provisions AWS infrastructure using Terraform for staging environment
# Usage: ./01-provision-infrastructure.sh

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TERRAFORM_DIR="/home/kp/novacron/deployments/terraform/dwcp-v3"
ENVIRONMENT="staging"
AWS_REGION="${AWS_REGION:-us-east-1}"
STATE_BUCKET="${STATE_BUCKET:-novacron-terraform-state}"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

check_prerequisites() {
    log "Checking prerequisites..."

    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed. Please install Terraform 1.5.0 or later."
        exit 1
    fi

    local tf_version=$(terraform version -json | jq -r '.terraform_version')
    log "Terraform version: $tf_version"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install AWS CLI v2."
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials are not configured. Please run 'aws configure'."
        exit 1
    fi

    local aws_account=$(aws sts get-caller-identity --query Account --output text)
    log "AWS Account: $aws_account"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        warning "kubectl is not installed. Kubernetes deployment will be skipped."
    fi

    success "Prerequisites check passed"
}

initialize_terraform() {
    log "Initializing Terraform..."

    cd "$TERRAFORM_DIR"

    # Initialize with S3 backend
    terraform init \
        -backend-config="bucket=${STATE_BUCKET}" \
        -backend-config="key=dwcp-v3/${ENVIRONMENT}/terraform.tfstate" \
        -backend-config="region=${AWS_REGION}" \
        -backend-config="encrypt=true" \
        -reconfigure

    if [ $? -eq 0 ]; then
        success "Terraform initialized successfully"
    else
        error "Terraform initialization failed"
        exit 1
    fi
}

create_tfvars() {
    log "Creating terraform.tfvars for staging environment..."

    cat > "$TERRAFORM_DIR/staging.tfvars" <<EOF
# DWCP v3 Staging Environment Configuration
environment = "staging"
project_name = "dwcp-v3-staging"
aws_region = "$AWS_REGION"

# VPC Configuration
vpc_cidr = "10.1.0.0/16"
public_subnet_cidrs = ["10.1.1.0/24", "10.1.2.0/24", "10.1.3.0/24"]
private_subnet_cidrs = ["10.1.10.0/24", "10.1.11.0/24", "10.1.12.0/24"]

# Network Configuration
allowed_cidr_blocks = ["0.0.0.0/0"]
monitoring_cidr_blocks = ["10.1.0.0/16"]
enable_nat_gateway = true
enable_https = false  # Use HTTP for staging

# Compute Configuration
instance_type = "t3.large"
min_size = 2
max_size = 6
desired_capacity = 3

# Logging
log_retention_days = 7

# Tags
common_tags = {
  Environment = "staging"
  Project     = "DWCP-v3"
  ManagedBy   = "Terraform"
  Phase       = "5-Staging-Deployment"
  CostCenter  = "Engineering"
}
EOF

    success "Created staging.tfvars"
}

validate_terraform() {
    log "Validating Terraform configuration..."

    cd "$TERRAFORM_DIR"
    terraform validate

    if [ $? -eq 0 ]; then
        success "Terraform configuration is valid"
    else
        error "Terraform validation failed"
        exit 1
    fi
}

plan_infrastructure() {
    log "Planning infrastructure changes..."

    cd "$TERRAFORM_DIR"
    terraform plan \
        -var-file="staging.tfvars" \
        -out="staging.tfplan"

    if [ $? -eq 0 ]; then
        success "Terraform plan completed"
        log "Plan saved to: staging.tfplan"
    else
        error "Terraform plan failed"
        exit 1
    fi
}

apply_infrastructure() {
    log "Applying infrastructure changes..."

    cd "$TERRAFORM_DIR"

    # Show plan summary
    terraform show -no-color staging.tfplan | head -50

    # Prompt for confirmation
    read -p "Do you want to apply this plan? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        warning "Infrastructure provisioning cancelled"
        exit 0
    fi

    # Apply plan
    terraform apply staging.tfplan

    if [ $? -eq 0 ]; then
        success "Infrastructure provisioned successfully"
    else
        error "Infrastructure provisioning failed"
        exit 1
    fi
}

save_outputs() {
    log "Saving Terraform outputs..."

    cd "$TERRAFORM_DIR"

    # Save all outputs to JSON
    terraform output -json > staging-outputs.json

    # Extract key outputs
    local vpc_id=$(terraform output -raw vpc_id 2>/dev/null || echo "N/A")
    local alb_dns=$(terraform output -raw alb_dns_name 2>/dev/null || echo "N/A")
    local sg_id=$(terraform output -raw security_group_id 2>/dev/null || echo "N/A")

    log "VPC ID: $vpc_id"
    log "ALB DNS: $alb_dns"
    log "Security Group: $sg_id"

    # Save to environment file
    cat > "$TERRAFORM_DIR/../staging-env.sh" <<EOF
#!/usr/bin/env bash
# DWCP v3 Staging Environment Variables
export DWCP_ENVIRONMENT="staging"
export DWCP_VPC_ID="$vpc_id"
export DWCP_ALB_DNS="$alb_dns"
export DWCP_SECURITY_GROUP="$sg_id"
export DWCP_AWS_REGION="$AWS_REGION"
EOF

    chmod +x "$TERRAFORM_DIR/../staging-env.sh"
    success "Outputs saved to staging-outputs.json and staging-env.sh"
}

verify_infrastructure() {
    log "Verifying infrastructure..."

    cd "$TERRAFORM_DIR"

    # Check VPC
    local vpc_id=$(terraform output -raw vpc_id 2>/dev/null)
    if aws ec2 describe-vpcs --vpc-ids "$vpc_id" &> /dev/null; then
        success "VPC verified: $vpc_id"
    else
        error "VPC verification failed"
        exit 1
    fi

    # Check ALB
    local alb_dns=$(terraform output -raw alb_dns_name 2>/dev/null)
    log "ALB DNS: $alb_dns"
    log "ALB will be accessible once application is deployed"

    success "Infrastructure verification completed"
}

notify_completion() {
    log "Sending deployment notification..."

    if command -v npx &> /dev/null; then
        npx claude-flow@alpha hooks notify \
            --message "DWCP v3 staging infrastructure provisioned successfully" \
            2>/dev/null || true
    fi

    # Save to memory
    if command -v npx &> /dev/null; then
        npx claude-flow@alpha hooks post-edit \
            --file "$TERRAFORM_DIR/staging-outputs.json" \
            --memory-key "swarm/phase5/staging/infrastructure" \
            2>/dev/null || true
    fi
}

main() {
    log "===== DWCP v3 Phase 5: Staging Infrastructure Provisioning ====="
    log "Environment: $ENVIRONMENT"
    log "AWS Region: $AWS_REGION"
    log ""

    check_prerequisites
    initialize_terraform
    create_tfvars
    validate_terraform
    plan_infrastructure
    apply_infrastructure
    save_outputs
    verify_infrastructure
    notify_completion

    echo ""
    success "===== Infrastructure Provisioning Complete ====="
    log "Next steps:"
    log "  1. Source environment variables: source deployments/staging-env.sh"
    log "  2. Deploy application: ./02-deploy-application.sh"
    log "  3. Run validation: ./03-validate-deployment.sh"
}

# Run main function
main "$@"
