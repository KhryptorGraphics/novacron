# NovaCron Load Testing Makefile
# Comprehensive automation for load testing infrastructure with performance optimization

.PHONY: help install clean setup test test-api test-vm test-ws test-db test-federation benchmark stress soak monitor report cleanup optimize profile tune

# Configuration
ENVIRONMENT ?= local
CONCURRENT_USERS ?= 1000
TEST_DURATION ?= 10m
API_TARGET ?= http://localhost:8080
WS_TARGET ?= ws://localhost:8080
PARALLEL_TESTS ?= true
CLEANUP_AFTER ?= true

# Performance optimization configuration
OPTIMIZATION_LEVEL ?= standard
TUNING_ENVIRONMENT ?= staging
PROFILING_DURATION ?= 60

# Directories
SCRIPTS_DIR := scripts
SCENARIOS_DIR := scenarios
REPORTS_DIR := reports
MONITORING_DIR := monitoring

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m

# Default target
help: ## Show this help message
	@echo "$(CYAN)NovaCron Load Testing Suite$(NC)"
	@echo "$(CYAN)============================$(NC)"
	@echo ""
	@echo "$(PURPLE)Load Testing:$(NC)"
	@grep -E '^test.*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(PURPLE)Performance Optimization:$(NC)"
	@grep -E '^(optimize|profile|tune|benchmark).*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(PURPLE)Monitoring & Reporting:$(NC)"
	@grep -E '^(monitor|report|dashboard).*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(PURPLE)Environment & Cleanup:$(NC)"
	@grep -E '^(setup|teardown|clean|validate).*:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(PURPLE)Configuration:$(NC)"
	@echo "  ENVIRONMENT:      $(ENVIRONMENT)"
	@echo "  CONCURRENT_USERS: $(CONCURRENT_USERS)"
	@echo "  TEST_DURATION:    $(TEST_DURATION)"
	@echo "  API_TARGET:       $(API_TARGET)"
	@echo "  WS_TARGET:        $(WS_TARGET)"
	@echo "  PARALLEL_TESTS:   $(PARALLEL_TESTS)"
	@echo ""
	@echo "$(PURPLE)Examples:$(NC)"
	@echo "  make test ENVIRONMENT=staging CONCURRENT_USERS=2000"
	@echo "  make optimize-full API_TARGET=https://api.novacron.com"
	@echo "  make profile-deep PROFILING_DURATION=300"
	@echo "  make auto-tune TUNING_ENVIRONMENT=production"

install: ## Install dependencies and setup environment
	@echo "$(YELLOW)Installing load testing dependencies...$(NC)"
	@npm install
	@echo "$(YELLOW)Installing k6...$(NC)"
	@chmod +x $(SCRIPTS_DIR)/install-k6.sh
	@$(SCRIPTS_DIR)/install-k6.sh
	@echo "$(GREEN)Dependencies installed successfully$(NC)"

clean: ## Clean up test artifacts and reports
	@echo "$(YELLOW)Cleaning up test artifacts...$(NC)"
	@rm -rf $(REPORTS_DIR)/*.json $(REPORTS_DIR)/*.html $(REPORTS_DIR)/*.log
	@rm -rf $(REPORTS_DIR)/profiling $(REPORTS_DIR)/tuning $(REPORTS_DIR)/exports
	@rm -rf $(MONITORING_DIR)/*.pid $(MONITORING_DIR)/*.log
	@rm -rf node_modules/.cache
	@echo "$(GREEN)Cleanup completed$(NC)"

setup: install ## Setup monitoring stack with Docker Compose
	@echo "$(YELLOW)Setting up monitoring infrastructure...$(NC)"
	@docker-compose -f docker-compose.monitoring.yml up -d
	@echo "$(YELLOW)Waiting for services to start...$(NC)"
	@sleep 30
	@echo "$(GREEN)Monitoring stack is ready$(NC)"
	@echo "Access points:"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin123)"
	@echo "  InfluxDB:   http://localhost:8086"

teardown: ## Tear down monitoring stack
	@echo "$(YELLOW)Tearing down monitoring infrastructure...$(NC)"
	@docker-compose -f docker-compose.monitoring.yml down -v
	@echo "$(GREEN)Monitoring stack stopped$(NC)"

# Individual test scenarios
test-api: ## Run API load testing (1000+ concurrent users)
	@echo "$(BLUE)Running API Load Test...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/api-load-test-results.json \
		--vus $(CONCURRENT_USERS) \
		--duration $(TEST_DURATION) \
		$(SCENARIOS_DIR)/api-load-test.js

test-vm: ## Run VM management load testing (1000+ VMs)
	@echo "$(BLUE)Running VM Management Load Test...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/vm-management-test-results.json \
		$(SCENARIOS_DIR)/vm-management-test.js

test-ws: ## Run WebSocket stress testing (2000+ connections)
	@echo "$(BLUE)Running WebSocket Stress Test...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/websocket-stress-test-results.json \
		--vus 2000 \
		--duration 15m \
		$(SCENARIOS_DIR)/websocket-stress-test.js

test-db: ## Run database performance testing
	@echo "$(BLUE)Running Database Performance Test...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/database-performance-test-results.json \
		$(SCENARIOS_DIR)/database-performance-test.js

test-federation: ## Run multi-cloud federation load testing
	@echo "$(BLUE)Running Federation Load Test...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/federation-load-test-results.json \
		$(SCENARIOS_DIR)/federation-load-test.js

benchmark: ## Run performance benchmarking suite
	@echo "$(BLUE)Running Performance Benchmark Suite...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/benchmark-results.json \
		$(SCENARIOS_DIR)/benchmark-suite.js

stress: ## Run stress testing (high load scenarios)
	@echo "$(BLUE)Running Stress Test...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/stress-test-results.json \
		--vus 2000 \
		--duration 30m \
		$(SCENARIOS_DIR)/stress-test.js

soak: ## Run soak testing (2+ hour sustained load)
	@echo "$(BLUE)Running Soak Test (2+ hours)...$(NC)"
	@echo "$(YELLOW)Warning: This test will run for 2+ hours$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/soak-test-results.json \
		--vus 200 \
		--duration 2h \
		$(SCENARIOS_DIR)/soak-test.js

# Performance Optimization Targets
optimize-analyze: ## Analyze performance from latest test results
	@echo "$(PURPLE)Analyzing performance metrics...$(NC)"
	@if [ -f $(REPORTS_DIR)/api-load-test-results.json ]; then \
		node $(SCRIPTS_DIR)/performance-optimizer.js analyze $(REPORTS_DIR)/api-load-test-results.json; \
	else \
		echo "$(RED)No test results found. Run 'make test-api' first$(NC)"; \
		exit 1; \
	fi

optimize-monitor: ## Start real-time performance monitoring
	@echo "$(PURPLE)Starting real-time performance monitoring...$(NC)"
	@node $(SCRIPTS_DIR)/performance-optimizer.js monitor $(PROFILING_DURATION)000

optimize-config: ## Optimize test configuration for current system
	@echo "$(PURPLE)Optimizing test configuration...$(NC)"
	@node $(SCRIPTS_DIR)/performance-optimizer.js optimize-config

optimize-baseline: ## Set performance baseline from test results
	@echo "$(PURPLE)Setting performance baseline...$(NC)"
	@if [ -f $(REPORTS_DIR)/api-load-test-results.json ]; then \
		node $(SCRIPTS_DIR)/performance-optimizer.js baseline $(REPORTS_DIR)/api-load-test-results.json; \
	else \
		echo "$(RED)No test results found. Run 'make test-api' first$(NC)"; \
		exit 1; \
	fi

# Advanced Profiling Targets
profile-quick: ## Quick performance profiling (30 seconds)
	@echo "$(PURPLE)Running quick performance profiling...$(NC)"
	@node $(SCRIPTS_DIR)/profiler.js profile $(API_TARGET) 30000

profile-standard: ## Standard performance profiling (60 seconds)
	@echo "$(PURPLE)Running standard performance profiling...$(NC)"
	@node $(SCRIPTS_DIR)/profiler.js profile $(API_TARGET) 60000

profile-deep: ## Deep performance profiling (5 minutes)
	@echo "$(PURPLE)Running deep performance profiling...$(NC)"
	@node $(SCRIPTS_DIR)/profiler.js profile $(API_TARGET) 300000

profile-bottlenecks: ## Analyze bottlenecks from latest profiling session
	@echo "$(PURPLE)Analyzing performance bottlenecks...$(NC)"
	@LATEST_PROFILE=$$(ls -t $(REPORTS_DIR)/profiling/profiling-session-*.json 2>/dev/null | head -1); \
	if [ -n "$$LATEST_PROFILE" ]; then \
		node $(SCRIPTS_DIR)/profiler.js bottlenecks "$$LATEST_PROFILE"; \
	else \
		echo "$(RED)No profiling sessions found. Run 'make profile-standard' first$(NC)"; \
	fi

# Auto-Tuning Targets
auto-tune: ## Run complete auto-tuning process for environment
	@echo "$(PURPLE)Starting auto-tuning for $(TUNING_ENVIRONMENT)...$(NC)"
	@node $(SCRIPTS_DIR)/performance-tuner.js auto-tune $(TUNING_ENVIRONMENT)

tune-quick: ## Quick performance tuning analysis
	@echo "$(PURPLE)Running quick tuning analysis...$(NC)"
	@node $(SCRIPTS_DIR)/performance-tuner.js quick-tune $(TUNING_ENVIRONMENT)

tune-config: ## Generate optimized configuration for environment
	@echo "$(PURPLE)Generating optimized configuration...$(NC)"
	@CPU_CORES=$$(nproc); \
	MEMORY_MB=$$(free -m | grep Mem | awk '{print $$2}'); \
	node $(SCRIPTS_DIR)/performance-tuner.js optimize-config $(TUNING_ENVIRONMENT) $$CPU_CORES $$MEMORY_MB

# Continuous Monitoring Targets
monitor-continuous: ## Start continuous performance monitoring
	@echo "$(PURPLE)Starting continuous performance monitoring...$(NC)"
	@node $(SCRIPTS_DIR)/continuous-monitoring.js start $(API_TARGET)

monitor-analyze: ## Analyze continuous monitoring data
	@echo "$(PURPLE)Analyzing continuous monitoring data...$(NC)"
	@LATEST_MONITOR=$$(ls -t $(REPORTS_DIR)/continuous-monitoring-*.json 2>/dev/null | head -1); \
	if [ -n "$$LATEST_MONITOR" ]; then \
		node $(SCRIPTS_DIR)/continuous-monitoring.js analyze "$$LATEST_MONITOR"; \
	else \
		echo "$(RED)No monitoring data found. Run 'make monitor-continuous' first$(NC)"; \
	fi

# Benchmark Comparison Targets
compare-results: ## Compare multiple benchmark results
	@echo "$(PURPLE)Comparing benchmark results...$(NC)"
	@RESULT_FILES=$$(ls -t $(REPORTS_DIR)/*-results.json 2>/dev/null | head -3); \
	if [ -n "$$RESULT_FILES" ]; then \
		node $(SCRIPTS_DIR)/benchmark-comparator.js compare $$RESULT_FILES; \
	else \
		echo "$(RED)Not enough result files found. Run tests first$(NC)"; \
	fi

compare-trend: ## Analyze performance trends over multiple results
	@echo "$(PURPLE)Analyzing performance trends...$(NC)"
	@RESULT_FILES=$$(ls -t $(REPORTS_DIR)/*-results.json 2>/dev/null | head -5); \
	if [ $$(echo $$RESULT_FILES | wc -w) -ge 3 ]; then \
		node $(SCRIPTS_DIR)/benchmark-comparator.js trend $$RESULT_FILES; \
	else \
		echo "$(RED)Need at least 3 result files for trend analysis$(NC)"; \
	fi

# Comprehensive test suites
test: install ## Run all core load tests
	@echo "$(BLUE)Running comprehensive load test suite...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) \
	 CONCURRENT_USERS=$(CONCURRENT_USERS) \
	 TEST_DURATION=$(TEST_DURATION) \
	 API_TARGET=$(API_TARGET) \
	 WS_TARGET=$(WS_TARGET) \
	 PARALLEL_TESTS=$(PARALLEL_TESTS) \
	 CLEANUP_AFTER=$(CLEANUP_AFTER) \
	 $(SCRIPTS_DIR)/run-load-tests.sh

test-all: install ## Run all tests including stress and soak tests
	@echo "$(BLUE)Running complete test suite (including stress and soak)...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) \
	 INCLUDE_STRESS_TESTS=true \
	 INCLUDE_SOAK_TESTS=true \
	 PARALLEL_TESTS=$(PARALLEL_TESTS) \
	 $(SCRIPTS_DIR)/run-load-tests.sh

test-quick: install ## Run quick validation tests (reduced load and duration)
	@echo "$(BLUE)Running quick validation tests...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) \
	 CONCURRENT_USERS=100 \
	 TEST_DURATION=2m \
	 PARALLEL_TESTS=true \
	 $(SCRIPTS_DIR)/run-load-tests.sh

# Optimization Workflows
optimize-full: ## Complete optimization workflow (test → analyze → tune → retest)
	@echo "$(PURPLE)Starting full optimization workflow...$(NC)"
	@$(MAKE) setup
	@echo "$(BLUE)Step 1: Baseline measurement$(NC)"
	@$(MAKE) test-api
	@echo "$(BLUE)Step 2: Performance analysis$(NC)"
	@$(MAKE) optimize-analyze
	@echo "$(BLUE)Step 3: Deep profiling$(NC)"
	@$(MAKE) profile-standard
	@echo "$(BLUE)Step 4: Auto-tuning$(NC)"
	@$(MAKE) auto-tune
	@echo "$(BLUE)Step 5: Validation test$(NC)"
	@$(MAKE) test-api
	@echo "$(BLUE)Step 6: Final analysis$(NC)"
	@$(MAKE) compare-results
	@$(MAKE) report
	@echo "$(GREEN)Full optimization workflow completed$(NC)"

optimize-quick: ## Quick optimization workflow (test → analyze → recommendations)
	@echo "$(PURPLE)Starting quick optimization workflow...$(NC)"
	@$(MAKE) test-api
	@$(MAKE) optimize-analyze
	@$(MAKE) profile-quick
	@echo "$(GREEN)Quick optimization completed$(NC)"

optimize-continuous: ## Start continuous optimization monitoring
	@echo "$(PURPLE)Starting continuous optimization monitoring...$(NC)"
	@$(MAKE) setup
	@$(MAKE) monitor-continuous &
	@sleep 5
	@echo "$(GREEN)Continuous optimization monitoring started$(NC)"
	@echo "$(YELLOW)Use 'make stop' to stop monitoring$(NC)"

# Performance Profiling Workflows
profile-comprehensive: ## Comprehensive profiling with all metrics
	@echo "$(PURPLE)Running comprehensive performance profiling...$(NC)"
	@$(MAKE) setup
	@$(MAKE) profile-deep
	@$(MAKE) optimize-monitor &
	@MONITOR_PID=$$!; \
	sleep $(PROFILING_DURATION); \
	kill $$MONITOR_PID || true; \
	$(MAKE) profile-bottlenecks
	@echo "$(GREEN)Comprehensive profiling completed$(NC)"

profile-regression: ## Profile for performance regression detection
	@echo "$(PURPLE)Running regression profiling...$(NC)"
	@$(MAKE) optimize-baseline || true
	@$(MAKE) test-api
	@$(MAKE) optimize-analyze
	@$(MAKE) compare-results
	@echo "$(GREEN)Regression profiling completed$(NC)"

# Auto-Tuning Workflows
tune-development: ## Auto-tune for development environment
	@$(MAKE) auto-tune TUNING_ENVIRONMENT=development

tune-staging: ## Auto-tune for staging environment
	@$(MAKE) auto-tune TUNING_ENVIRONMENT=staging

tune-production: ## Auto-tune for production environment (with confirmation)
	@echo "$(RED)WARNING: Auto-tuning production environment$(NC)"
	@echo "$(YELLOW)This will run performance tests against production. Continue? [y/N]$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		$(MAKE) auto-tune TUNING_ENVIRONMENT=production; \
	else \
		echo "Production auto-tuning cancelled"; \
	fi

# Monitoring and reporting
monitor: ## Start monitoring services
	@echo "$(BLUE)Starting load test monitoring...$(NC)"
	@node $(SCRIPTS_DIR)/start-monitoring.js &
	@echo "Monitoring started. Press Ctrl+C to stop."

monitor-dashboard: ## Open monitoring dashboard
	@echo "$(BLUE)Opening monitoring dashboard...$(NC)"
	@echo "Grafana Dashboard: http://localhost:3000"
	@echo "Prometheus UI: http://localhost:9090"
	@echo "InfluxDB UI: http://localhost:8086"

dashboard-performance: ## Open performance optimization dashboard
	@echo "$(PURPLE)Performance optimization dashboard:$(NC)"
	@echo "Real-time metrics: http://localhost:3000/d/performance"
	@echo "Profiling data: $(REPORTS_DIR)/profiling/"
	@echo "Tuning reports: $(REPORTS_DIR)/tuning/"

report: ## Generate comprehensive performance reports
	@echo "$(BLUE)Generating load test reports...$(NC)"
	@node $(SCRIPTS_DIR)/generate-report.js
	@echo "$(GREEN)Reports generated in $(REPORTS_DIR)/$(NC)"

report-optimization: ## Generate optimization-focused reports
	@echo "$(PURPLE)Generating optimization reports...$(NC)"
	@if [ -f $(REPORTS_DIR)/api-load-test-results.json ]; then \
		$(MAKE) optimize-analyze; \
	fi
	@if [ -d $(REPORTS_DIR)/profiling ]; then \
		$(MAKE) profile-bottlenecks; \
	fi
	@if [ -d $(REPORTS_DIR)/tuning ]; then \
		echo "$(GREEN)Tuning reports available in $(REPORTS_DIR)/tuning/$(NC)"; \
	fi

# Cleanup and maintenance
cleanup: ## Cleanup test data and artifacts
	@echo "$(BLUE)Cleaning up test data...$(NC)"
	@node $(SCRIPTS_DIR)/cleanup-test-data.js

cleanup-dry-run: ## Show what would be cleaned up (dry run)
	@echo "$(BLUE)Cleanup dry run...$(NC)"
	@DRY_RUN=true node $(SCRIPTS_DIR)/cleanup-test-data.js --dry-run

cleanup-force: ## Force cleanup all test artifacts
	@echo "$(YELLOW)Force cleaning all test artifacts...$(NC)"
	@$(MAKE) clean
	@node $(SCRIPTS_DIR)/cleanup-test-data.js
	@docker-compose -f docker-compose.monitoring.yml down -v 2>/dev/null || true
	@echo "$(GREEN)Force cleanup completed$(NC)"

# Validation and verification
validate: ## Validate test environment and configuration
	@echo "$(BLUE)Validating test environment...$(NC)"
	@$(SCRIPTS_DIR)/run-load-tests.sh --validate-only
	@echo "$(GREEN)Environment validation completed$(NC)"

validate-optimization: ## Validate optimization improvements
	@echo "$(PURPLE)Validating optimization improvements...$(NC)"
	@if [ -f $(REPORTS_DIR)/baseline-metrics.json ]; then \
		$(MAKE) test-api; \
		$(MAKE) optimize-analyze; \
		$(MAKE) compare-results; \
	else \
		echo "$(YELLOW)No baseline found. Setting baseline...$(NC)"; \
		$(MAKE) test-api; \
		$(MAKE) optimize-baseline; \
	fi

check-config: ## Check test configuration
	@echo "$(BLUE)Test Configuration:$(NC)"
	@echo "  Environment:      $(ENVIRONMENT)"
	@echo "  Concurrent Users: $(CONCURRENT_USERS)"
	@echo "  Test Duration:    $(TEST_DURATION)"
	@echo "  API Target:       $(API_TARGET)"
	@echo "  WebSocket Target: $(WS_TARGET)"
	@echo "  Parallel Tests:   $(PARALLEL_TESTS)"
	@echo "  Cleanup After:    $(CLEANUP_AFTER)"
	@echo ""
	@echo "$(PURPLE)Optimization Configuration:$(NC)"
	@echo "  Optimization Level: $(OPTIMIZATION_LEVEL)"
	@echo "  Tuning Environment: $(TUNING_ENVIRONMENT)"
	@echo "  Profiling Duration: $(PROFILING_DURATION)s"

# Environment-specific targets
test-local: ## Run tests against local environment
	@$(MAKE) test ENVIRONMENT=local API_TARGET=http://localhost:8080

test-staging: ## Run tests against staging environment
	@$(MAKE) test ENVIRONMENT=staging API_TARGET=https://staging.novacron.com

test-production: ## Run tests against production environment (use with caution)
	@echo "$(RED)WARNING: Testing against production environment$(NC)"
	@echo "$(YELLOW)This will generate significant load. Continue? [y/N]$(NC)"
	@read -r response; \
	if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
		$(MAKE) test ENVIRONMENT=production API_TARGET=https://api.novacron.com; \
	else \
		echo "Production test cancelled"; \
	fi

# CI/CD Integration targets
ci-test: ## Run tests suitable for CI/CD pipeline
	@echo "$(BLUE)Running CI/CD pipeline tests...$(NC)"
	@$(MAKE) test-quick PARALLEL_TESTS=true CLEANUP_AFTER=true

ci-benchmark: ## Run benchmark tests for CI/CD
	@echo "$(BLUE)Running CI/CD benchmark tests...$(NC)"
	@ENVIRONMENT=$(ENVIRONMENT) k6 run \
		--out json=$(REPORTS_DIR)/ci-benchmark-results.json \
		--vus 50 \
		--duration 5m \
		$(SCENARIOS_DIR)/benchmark-suite.js

ci-regression: ## Run regression tests for CI/CD
	@echo "$(BLUE)Running CI/CD regression tests...$(NC)"
	@$(MAKE) test-api CONCURRENT_USERS=100 TEST_DURATION=3m
	@$(MAKE) optimize-analyze
	@$(MAKE) compare-results || echo "$(YELLOW)No baseline for comparison$(NC)"

# Utility targets
logs: ## Show recent test logs
	@echo "$(BLUE)Recent test logs:$(NC)"
	@if [ -f $(REPORTS_DIR)/test-execution.log ]; then \
		tail -50 $(REPORTS_DIR)/test-execution.log; \
	else \
		echo "No test logs found"; \
	fi

status: ## Show current test status
	@echo "$(BLUE)Load Test Status:$(NC)"
	@if pgrep -f "k6 run" > /dev/null; then \
		echo "$(GREEN)✓$(NC) Load tests are running"; \
		echo "Active k6 processes:"; \
		pgrep -f "k6 run" | xargs ps -p; \
	else \
		echo "$(YELLOW)○$(NC) No active load tests"; \
	fi
	@if [ -f $(MONITORING_DIR)/monitoring.pid ]; then \
		echo "$(GREEN)✓$(NC) Monitoring is active"; \
	else \
		echo "$(YELLOW)○$(NC) Monitoring is not running"; \
	fi
	@if pgrep -f "continuous-monitoring" > /dev/null; then \
		echo "$(GREEN)✓$(NC) Continuous monitoring is active"; \
	else \
		echo "$(YELLOW)○$(NC) Continuous monitoring is not running"; \
	fi

stop: ## Stop all running tests and monitoring
	@echo "$(YELLOW)Stopping all load tests and monitoring...$(NC)"
	@pkill -f "k6 run" || true
	@pkill -f "continuous-monitoring" || true
	@pkill -f "performance-optimizer" || true
	@pkill -f "profiler.js" || true
	@if [ -f $(MONITORING_DIR)/monitoring.pid ]; then \
		kill $(shell cat $(MONITORING_DIR)/monitoring.pid) || true; \
		rm -f $(MONITORING_DIR)/monitoring.pid; \
	fi
	@echo "$(GREEN)All tests and monitoring stopped$(NC)"

# Development targets
dev-setup: ## Setup development environment for load testing
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@$(MAKE) install
	@$(MAKE) setup
	@echo "$(GREEN)Development environment ready$(NC)"

dev-test: ## Run development tests with reduced load
	@echo "$(BLUE)Running development load tests...$(NC)"
	@$(MAKE) test CONCURRENT_USERS=50 TEST_DURATION=2m PARALLEL_TESTS=false

dev-optimize: ## Development optimization workflow
	@echo "$(PURPLE)Running development optimization...$(NC)"
	@$(MAKE) dev-test
	@$(MAKE) optimize-analyze
	@$(MAKE) profile-quick
	@echo "$(GREEN)Development optimization completed$(NC)"

# Export and integration targets
export-prometheus: ## Export metrics for Prometheus
	@echo "$(BLUE)Exporting metrics for Prometheus...$(NC)"
	@if [ -f $(REPORTS_DIR)/continuous-monitoring-*.json ]; then \
		LATEST_MONITOR=$$(ls -t $(REPORTS_DIR)/continuous-monitoring-*.json | head -1); \
		node -e "const data = require('./$$LATEST_MONITOR'); console.log('# Exported metrics'); Object.entries(data.rawMetrics[data.rawMetrics.length-1].summary).forEach(([k,v]) => console.log(\`novacron_\${k} \${v}\`))"; \
	else \
		echo "$(RED)No monitoring data to export$(NC)"; \
	fi

export-csv: ## Export results to CSV format
	@echo "$(BLUE)Exporting results to CSV...$(NC)"
	@node -e "
		const fs = require('fs'); 
		const files = fs.readdirSync('$(REPORTS_DIR)').filter(f => f.endsWith('-results.json')); 
		files.forEach(f => {
			const data = JSON.parse(fs.readFileSync('$(REPORTS_DIR)/' + f, 'utf8'));
			const metrics = data.metrics;
			const csv = 'metric,value\n' + Object.entries(metrics).map(([k,v]) => k + ',' + (v.avg || v.rate || v.count || v.max || v)).join('\n');
			fs.writeFileSync('$(REPORTS_DIR)/' + f.replace('.json', '.csv'), csv);
		});
		console.log('$(GREEN)CSV exports completed$(NC)');
	"

# Advanced workflows
performance-suite: ## Complete performance testing and optimization suite
	@echo "$(PURPLE)Starting complete performance suite...$(NC)"
	@$(MAKE) setup
	@echo "$(BLUE)Phase 1: Baseline establishment$(NC)"
	@$(MAKE) test CONCURRENT_USERS=500 TEST_DURATION=5m
	@$(MAKE) optimize-baseline
	@echo "$(BLUE)Phase 2: Deep profiling$(NC)"
	@$(MAKE) profile-deep
	@echo "$(BLUE)Phase 3: Auto-tuning$(NC)"
	@$(MAKE) auto-tune
	@echo "$(BLUE)Phase 4: Validation testing$(NC)"
	@$(MAKE) test CONCURRENT_USERS=$(CONCURRENT_USERS) TEST_DURATION=$(TEST_DURATION)
	@echo "$(BLUE)Phase 5: Regression analysis$(NC)"
	@$(MAKE) compare-results
	@echo "$(BLUE)Phase 6: Comprehensive reporting$(NC)"
	@$(MAKE) report
	@$(MAKE) report-optimization
	@$(MAKE) teardown
	@echo "$(GREEN)Complete performance suite finished$(NC)"

regression-detection: ## Continuous regression detection workflow
	@echo "$(PURPLE)Starting regression detection...$(NC)"
	@$(MAKE) test-api CONCURRENT_USERS=200 TEST_DURATION=3m
	@$(MAKE) optimize-analyze
	@$(MAKE) compare-results
	@echo "$(GREEN)Regression detection completed$(NC)"

# Quick reference targets
quick: test-quick ## Alias for test-quick
all: test-all ## Alias for test-all
full: performance-suite ## Alias for performance-suite
optimize: optimize-full ## Alias for optimize-full
profile: profile-standard ## Alias for profile-standard
tune: auto-tune ## Alias for auto-tune

# Advanced utility targets
health-check: ## Check system health before load testing
	@echo "$(BLUE)Checking system health...$(NC)"
	@echo "CPU Cores: $$(nproc)"
	@echo "Memory: $$(free -h | grep Mem | awk '{print $$2}')"
	@echo "Disk Space: $$(df -h . | tail -1 | awk '{print $$4}') available"
	@echo "Network: $$(ping -c 1 8.8.8.8 > /dev/null && echo 'OK' || echo 'FAILED')"
	@echo "API Target: $$(curl -s $(API_TARGET)/api/cluster/health > /dev/null && echo 'OK' || echo 'UNREACHABLE')"
	@echo "K6 Version: $$(k6 version | head -1)"

performance-check: ## Quick performance health check
	@echo "$(PURPLE)Checking performance health...$(NC)"
	@$(MAKE) health-check
	@$(MAKE) test-api CONCURRENT_USERS=50 TEST_DURATION=1m
	@$(MAKE) optimize-analyze
	@echo "$(GREEN)Performance health check completed$(NC)"

# Emergency targets
emergency-stop: ## Emergency stop all processes and cleanup
	@echo "$(RED)EMERGENCY STOP - Killing all processes...$(NC)"
	@pkill -9 -f "k6" || true
	@pkill -9 -f "node.*monitoring" || true
	@pkill -9 -f "node.*profiler" || true
	@pkill -9 -f "node.*optimizer" || true
	@docker-compose -f docker-compose.monitoring.yml kill || true
	@echo "$(GREEN)Emergency stop completed$(NC)"