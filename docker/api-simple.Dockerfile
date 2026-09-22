# Simple single-stage Docker build for the canonical NovaCron API server
FROM golang:1.25-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod and sum files (sdk/ is a module-replacement target of go.mod)
COPY go.mod go.sum ./
COPY sdk/go/go.mod sdk/go/go.sum ./sdk/go/

# Copy backend directory first (needed for module replacement)
COPY sdk ./sdk
COPY backend ./backend

# Download dependencies
RUN go mod download

# Build the canonical API server
RUN CGO_ENABLED=0 GOOS=linux go build -a -o api-server-simple ./backend/cmd/api-server

# Production stage
FROM scratch

# Copy timezone data and certificates from builder
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the binary
COPY --from=builder /app/api-server-simple /api-server

# Set environment variables
ENV TZ=UTC

# Expose port
EXPOSE 8090

# Run the server
ENTRYPOINT ["/api-server"]
