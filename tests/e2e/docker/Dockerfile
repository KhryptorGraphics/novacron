# Multi-stage Dockerfile for E2E test environment

# Stage 1: Base image with system dependencies
FROM mcr.microsoft.com/playwright:v1.40.0-jammy AS base

# Set working directory
WORKDIR /app

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    curl \
    jq \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy package files
COPY package*.json ./
COPY tests/e2e/package*.json ./tests/e2e/

# Install Node.js dependencies
RUN npm ci && \
    cd tests/e2e && npm ci

# Install Playwright browsers (already in base image, but ensure latest)
RUN npx playwright install --with-deps

# Stage 3: Application build
FROM dependencies AS builder

# Copy application source
COPY . .

# Build application
RUN npm run build

# Stage 4: Test runtime environment
FROM base AS test-runtime

# Copy dependencies from dependencies stage
COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=dependencies /app/tests/e2e/node_modules ./tests/e2e/node_modules

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/build ./build

# Copy test files and configurations
COPY tests/e2e ./tests/e2e
COPY playwright.config.ts ./
COPY tsconfig.json ./

# Copy scripts
COPY tests/e2e/scripts ./tests/e2e/scripts
RUN chmod +x tests/e2e/scripts/*.sh

# Environment variables
ENV CI=true
ENV TEST_ENV=docker
ENV NODE_ENV=test
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Default command
CMD ["npm", "run", "test:e2e"]

# Stage 5: Debug image with additional tools
FROM test-runtime AS debug

# Install debugging tools
RUN apt-get update && apt-get install -y \
    vim \
    tmux \
    htop \
    net-tools \
    iputils-ping \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Keep container running for debugging
CMD ["tail", "-f", "/dev/null"]
