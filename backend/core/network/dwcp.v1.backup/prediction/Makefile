.PHONY: help install test train deploy clean

# Default target
help:
	@echo "PBA (Predictive Bandwidth Allocation) Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  benchmark    - Run benchmarks"
	@echo "  export-data  - Export training data from Prometheus"
	@echo "  train        - Train LSTM model"
	@echo "  train-docker - Train using Docker (GPU)"
	@echo "  deploy       - Deploy trained model"
	@echo "  validate     - Validate model performance"
	@echo "  monitor      - Start monitoring stack"
	@echo "  clean        - Clean build artifacts"

# Install ONNX Runtime and dependencies
install:
	@echo "Installing ONNX Runtime..."
	@if [ ! -f /usr/local/lib/libonnxruntime.so ]; then \
		wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.0/onnxruntime-linux-x64-1.15.0.tgz && \
		tar -xzf onnxruntime-linux-x64-1.15.0.tgz && \
		sudo cp onnxruntime-linux-x64-1.15.0/lib/* /usr/local/lib/ && \
		sudo ldconfig && \
		rm -rf onnxruntime-linux-x64-1.15.0*; \
	fi
	@echo "Installing Go dependencies..."
	go get github.com/yalue/onnxruntime_go
	go get github.com/prometheus/client_golang/prometheus
	@echo "Installing Python dependencies..."
	cd training && pip install -r requirements.txt
	@echo "Installation complete!"

# Run tests
test:
	@echo "Running tests..."
	go test -v -race -coverprofile=coverage.out .
	@echo "Coverage report:"
	go tool cover -func=coverage.out

# Run benchmarks
benchmark:
	@echo "Running benchmarks..."
	go test -bench=. -benchmem -benchtime=10s .

# Export training data from Prometheus
export-data:
	@echo "Exporting training data from Prometheus..."
	cd training && \
	export PROMETHEUS_URL=${PROMETHEUS_URL:-http://localhost:9090} && \
	export OUTPUT_FILE=${OUTPUT_FILE:-./training_data/training_data.csv} && \
	export DAYS_BACK=${DAYS_BACK:-30} && \
	bash export_metrics.sh
	@echo "Data exported to training/training_data/training_data.csv"

# Train model locally
train: export-data
	@echo "Training LSTM model..."
	cd training && \
	mkdir -p models && \
	python3 train_lstm.py \
		--data training_data/training_data.csv \
		--output models \
		--epochs ${EPOCHS:-100} \
		--batch-size ${BATCH_SIZE:-32} \
		--learning-rate ${LR:-0.001}
	@echo "Training complete! Model saved to training/models/"

# Train using Docker with GPU support
train-docker:
	@echo "Training with Docker (GPU)..."
	cd training && \
	docker-compose --profile training up trainer
	@echo "Training complete! Model saved to training/models/"

# Deploy trained model
deploy:
	@echo "Deploying trained model..."
	@LATEST_MODEL=$$(ls -t training/models/bandwidth_lstm_*.onnx | head -1) && \
	sudo mkdir -p /var/lib/novacron/dwcp/models && \
	sudo cp $$LATEST_MODEL /var/lib/novacron/dwcp/models/bandwidth_lstm_v1.onnx && \
	sudo cp training/models/model_metadata_*.json /var/lib/novacron/dwcp/models/ && \
	echo "Model deployed to /var/lib/novacron/dwcp/models/"

# Validate model performance
validate:
	@echo "Validating model performance..."
	@python3 -c "import json; \
		metrics = json.load(open('training/models/model_metadata_v*.json')); \
		print('Model Performance:'); \
		print(f\"  Bandwidth MAE: {metrics['performance_metrics']['bandwidth_mae']} Mbps\"); \
		print(f\"  Latency MAE: {metrics['performance_metrics']['latency_mae']} ms\"); \
		print(f\"  Overall Accuracy: {metrics['performance_metrics']['overall_accuracy']*100:.1f}%\"); \
		assert metrics['performance_metrics']['overall_accuracy'] > 0.85, 'Accuracy too low!'; \
		print('✓ Model meets performance requirements')"

# Start monitoring stack
monitor:
	@echo "Starting monitoring stack..."
	cd training && docker-compose up -d prometheus grafana
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

# Stop monitoring stack
monitor-stop:
	@echo "Stopping monitoring stack..."
	cd training && docker-compose down

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f coverage.out
	rm -rf training/models/*
	rm -rf training/training_data/*
	rm -rf training/logs/*
	@echo "Clean complete!"

# Run example integration
example:
	@echo "Running example integration..."
	go run example_integration.go

# Check model status
status:
	@echo "PBA System Status:"
	@echo ""
	@if [ -f /var/lib/novacron/dwcp/models/bandwidth_lstm_v1.onnx ]; then \
		echo "✓ Model deployed"; \
		ls -lh /var/lib/novacron/dwcp/models/bandwidth_lstm_v1.onnx; \
	else \
		echo "✗ Model not deployed"; \
	fi
	@echo ""
	@if [ -f /var/lib/novacron/dwcp/network_samples.jsonl ]; then \
		echo "✓ Data collection active"; \
		SAMPLES=$$(wc -l < /var/lib/novacron/dwcp/network_samples.jsonl); \
		echo "  Samples collected: $$SAMPLES"; \
	else \
		echo "✗ No data collected yet"; \
	fi

# Quick start - full pipeline
quickstart: install export-data train deploy validate
	@echo ""
	@echo "Quick start complete!"
	@echo "PBA system is ready to use."
	@echo ""
	@echo "Next steps:"
	@echo "  1. Start your Go application with PBA integration"
	@echo "  2. Monitor predictions: make monitor"
	@echo "  3. Check status: make status"

# Continuous training (runs daily)
continuous-train:
	@echo "Setting up continuous training..."
	@echo "0 2 * * * cd $$(pwd) && make train deploy" | crontab -
	@echo "Continuous training scheduled for 2 AM daily"

# Performance report
report:
	@echo "Generating performance report..."
	@go test -bench=. -benchmem -cpuprofile=cpu.prof -memprofile=mem.prof .
	@echo ""
	@echo "CPU Profile:"
	@go tool pprof -top -cum cpu.prof | head -20
	@echo ""
	@echo "Memory Profile:"
	@go tool pprof -top -cum mem.prof | head -20

# Live monitoring
live:
	@echo "Starting live monitoring..."
	@watch -n 5 'curl -s http://localhost:9090/api/v1/query?query=dwcp_pba_prediction_accuracy | jq ".data.result[0].value[1]"'
