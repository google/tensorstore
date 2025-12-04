FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git curl python3 build-essential clang unzip zip ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    curl -L "https://github.com/bazelbuild/bazelisk/releases/download/v1.27.0/bazelisk-linux-${arch}" \
      -o /usr/local/bin/bazelisk; \
    chmod +x /usr/local/bin/bazelisk; \
    ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel
WORKDIR /workspace
