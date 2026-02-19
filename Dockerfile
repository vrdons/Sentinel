# Builder stage
FROM rust:1.82-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    git \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Build the application
RUN cargo build --release

# Final stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/sentinel /usr/local/bin/sentinel
# Copy templates as they are needed at runtime by Askama
COPY --from=builder /app/templates /app/templates

# Expose proxy and dashboard ports
EXPOSE 8080 3000

# Default command
CMD ["sentinel", "start"]
