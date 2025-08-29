# Enhanced NovaCron Kubernetes Operator Makefile

# Image settings
IMG_NAME = novacron/k8s-operator
IMG_TAG = v2.0.0-enhanced
IMG = $(IMG_NAME):$(IMG_TAG)

# Kubernetes settings
NAMESPACE = novacron-system
KUBECONFIG ?= ~/.kube/config

# Go settings
GO_VERSION = 1.21
GOOS = linux
GOARCH = amd64

# Tools
KUBECTL = kubectl
DOCKER = docker
GO = go

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: fmt
fmt: ## Format Go code
	$(GO) fmt ./...

.PHONY: vet
vet: ## Run go vet
	$(GO) vet ./...

.PHONY: test
test: ## Run tests
	$(GO) test -v ./pkg/...

.PHONY: lint
lint: ## Run golangci-lint
	@if command -v golangci-lint > /dev/null; then \
		golangci-lint run; \
	else \
		echo "golangci-lint not installed, skipping lint"; \
	fi

.PHONY: generate
generate: ## Generate code (if needed)
	$(GO) generate ./...

##@ Build

.PHONY: build
build: ## Build the operator binary
	CGO_ENABLED=0 GOOS=$(GOOS) GOARCH=$(GOARCH) $(GO) build \
		-ldflags='-w -s -extldflags "-static"' \
		-a -installsuffix cgo \
		-o bin/manager cmd/manager/main.go

.PHONY: docker-build
docker-build: ## Build Docker image
	$(DOCKER) build -t $(IMG) .

.PHONY: docker-push
docker-push: ## Push Docker image
	$(DOCKER) push $(IMG)

.PHONY: docker-build-push
docker-build-push: docker-build docker-push ## Build and push Docker image

##@ Deployment

.PHONY: deploy-crds
deploy-crds: ## Deploy CRDs only
	$(KUBECTL) apply -f deploy/crds/

.PHONY: deploy-rbac
deploy-rbac: ## Deploy RBAC only
	$(KUBECTL) apply -f deploy/rbac/

.PHONY: deploy
deploy: ## Deploy the operator
	./deploy/deploy-enhanced-operator.sh

.PHONY: deploy-examples
deploy-examples: ## Deploy with examples
	DEPLOY_EXAMPLES=true ./deploy/deploy-enhanced-operator.sh --examples

.PHONY: undeploy
undeploy: ## Remove the operator
	$(KUBECTL) delete -f deploy/operator.yaml --ignore-not-found=true
	$(KUBECTL) delete -f deploy/rbac/ --ignore-not-found=true
	$(KUBECTL) delete -f deploy/crds/ --ignore-not-found=true
	$(KUBECTL) delete namespace $(NAMESPACE) --ignore-not-found=true

##@ Development Environment

.PHONY: install-tools
install-tools: ## Install development tools
	@echo "Installing development tools..."
	@if ! command -v golangci-lint > /dev/null; then \
		curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(shell go env GOPATH)/bin v1.54.2; \
	fi
	@if ! command -v controller-gen > /dev/null; then \
		go install sigs.k8s.io/controller-tools/cmd/controller-gen@latest; \
	fi

.PHONY: mod-download
mod-download: ## Download Go modules
	$(GO) mod download
	$(GO) mod tidy

.PHONY: mod-verify
mod-verify: ## Verify Go modules
	$(GO) mod verify

##@ Testing

.PHONY: test-unit
test-unit: ## Run unit tests
	$(GO) test -v -race -coverprofile=coverage.out ./pkg/...

.PHONY: test-integration
test-integration: ## Run integration tests (requires cluster)
	@echo "Running integration tests..."
	$(GO) test -v -tags=integration ./test/integration/...

.PHONY: test-coverage
test-coverage: test-unit ## Generate test coverage report
	$(GO) tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

##@ Monitoring

.PHONY: deploy-monitoring
deploy-monitoring: ## Deploy monitoring stack
	$(KUBECTL) apply -f deploy/monitoring/

.PHONY: port-forward-metrics
port-forward-metrics: ## Port-forward operator metrics
	$(KUBECTL) port-forward -n $(NAMESPACE) svc/novacron-operator-metrics 8080:8080

.PHONY: logs
logs: ## Show operator logs
	$(KUBECTL) logs -f -n $(NAMESPACE) -l app=novacron-operator

.PHONY: status
status: ## Show operator status
	@echo "=== Operator Pods ==="
	$(KUBECTL) get pods -n $(NAMESPACE) -l app=novacron-operator -o wide
	@echo ""
	@echo "=== Operator Services ==="
	$(KUBECTL) get svc -n $(NAMESPACE)
	@echo ""
	@echo "=== CRDs ==="
	$(KUBECTL) get crds | grep novacron.io
	@echo ""
	@echo "=== Recent Events ==="
	$(KUBECTL) get events -n $(NAMESPACE) --sort-by='.lastTimestamp' | tail -10

##@ Examples

.PHONY: create-example-vm
create-example-vm: ## Create example VirtualMachine
	$(KUBECTL) apply -f deploy/examples/basic-vm.yaml

.PHONY: create-example-multicloud
create-example-multicloud: ## Create example MultiCloudVM
	$(KUBECTL) apply -f deploy/examples/multicloud-vm.yaml

.PHONY: create-example-ai-policy
create-example-ai-policy: ## Create example AI Scheduling Policy
	$(KUBECTL) apply -f deploy/examples/ai-scheduling-policy.yaml

.PHONY: create-example-cache
create-example-cache: ## Create example Cache Integration
	$(KUBECTL) apply -f deploy/examples/cache-integration.yaml

.PHONY: list-resources
list-resources: ## List all NovaCron resources
	@echo "=== VirtualMachines ==="
	$(KUBECTL) get virtualmachines --all-namespaces
	@echo ""
	@echo "=== MultiCloudVMs ==="
	$(KUBECTL) get multicloudvms --all-namespaces
	@echo ""
	@echo "=== AI Scheduling Policies ==="
	$(KUBECTL) get aischedulingpolicies --all-namespaces
	@echo ""
	@echo "=== Cache Integrations ==="
	$(KUBECTL) get cacheintegrations --all-namespaces

##@ Utility

.PHONY: clean
clean: ## Clean build artifacts
	rm -rf bin/
	rm -f coverage.out coverage.html

.PHONY: kind-cluster
kind-cluster: ## Create kind cluster for testing
	@if ! command -v kind > /dev/null; then \
		echo "kind not found, please install: https://kind.sigs.k8s.io/docs/user/quick-start/"; \
		exit 1; \
	fi
	kind create cluster --name novacron-test --config - <<EOF
	kind: Cluster
	apiVersion: kind.x-k8s.io/v1alpha4
	nodes:
	- role: control-plane
	- role: worker
	- role: worker
	EOF

.PHONY: kind-delete
kind-delete: ## Delete kind cluster
	kind delete cluster --name novacron-test

.PHONY: kind-load-image
kind-load-image: docker-build ## Load Docker image into kind cluster
	kind load docker-image $(IMG) --name novacron-test

##@ Release

.PHONY: tag
tag: ## Tag the current commit
	git tag -a $(IMG_TAG) -m "Release $(IMG_TAG)"
	git push origin $(IMG_TAG)

.PHONY: release
release: test lint docker-build-push ## Build, test and release
	@echo "Released $(IMG)"

.PHONY: all
all: fmt vet test build ## Run all checks and build

# Default target
.DEFAULT_GOAL := help