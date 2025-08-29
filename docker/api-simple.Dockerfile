# Simple single-stage Docker build for improved NovaCron API server
FROM golang:1.23-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy the specific main file and build it directly
COPY backend/cmd/api-server/main_improved.go .

# Initialize a simple module for this specific build
RUN go mod init novacron-api-simple && \
    go get github.com/gorilla/mux@latest

# Build the improved API server
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -a -installsuffix cgo \
    -o api-server-simple \
    main_improved.go

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