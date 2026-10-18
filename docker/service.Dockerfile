# Rust inference service. ONNX Runtime binaries are downloaded by the ort crate at build time and copied next to
# the binary (copy-dylibs); the runtime image is distroless-style Debian slim with a non-root user.
# trixie, not bookworm: the static ONNX Runtime that ort downloads needs glibc 2.38+ and GCC 13+ libstdc++.
FROM rust:1.95-slim-trixie AS build
# g++ provides libstdc++ for linking ONNX Runtime (C++), which the ort crate links into the binary.
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config ca-certificates g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY service ./service
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked -p pathology-service \
    && mkdir -p /out && cp target/release/pathology-service /out/ \
    && (cp target/release/libonnxruntime*.so* /out/ 2>/dev/null || true)

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /out/ /app/
ENV LD_LIBRARY_PATH=/app MODEL_PATH=/models/pcam/1/model.onnx MODEL_META_PATH=/models/pcam/1/model.json PORT=8080
USER app
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 CMD curl -fsS http://127.0.0.1:8080/ready || exit 1
ENTRYPOINT ["/app/pathology-service"]
