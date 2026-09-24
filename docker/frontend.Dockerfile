# Build stage
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Build args for optional runtime configuration (no defaults - same-origin fallback at runtime)
ARG NEXT_PUBLIC_API_URL
ARG NEXT_PUBLIC_WS_URL

# Expose build args as env vars for the build (empty if not provided)
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-} \
    NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL:-}

# Copy package.json and package-lock.json
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy source code
COPY frontend/ ./

# Build application
RUN npm run build

# Production stage
FROM node:18-alpine AS runner

# Set working directory
WORKDIR /app

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Set environment variables (runtime values take precedence over build-time)
ENV NODE_ENV=production \
    PORT=3000 \
    # NEXT_PUBLIC_API_URL and NEXT_PUBLIC_WS_URL are baked in at build time
    # from ARGs above. No defaults here - origin.ts falls back to same-origin.
    NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-} \
    NEXT_PUBLIC_WS_URL=${NEXT_PUBLIC_WS_URL:-}

# Copy build artifacts from builder stage.
# NOTE (novacron-5c7): frontend/next.config.js does not enable
# `output: 'standalone'`, so the standalone bundle never exists and the old
# COPY of /app/.next/standalone failed the build. Ship the full .next build
# with node_modules and run `next start` instead.
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next ./.next
COPY --from=builder --chown=nextjs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nextjs:nodejs /app/package.json ./package.json

# Set user
USER nextjs

# Health check: the app listens on PORT=3000 inside the container (the old
# probe hit localhost:8092/api/health — a route that does not exist in the
# frontend app, and on the wrong port); probe the root instead.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:3000/ || exit 1

# Expose port
EXPOSE 3000

# Set command
CMD ["npm", "start"]
