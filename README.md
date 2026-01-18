# vsearch

A vector search tool powered by Qdrant and ONNX Runtime.

## Usage

1. Set up [Qdrant server](https://qdrant.tech/).
2. Create a collection with appropriate vector size and distance metric. BGESmallZHV15 uses 512-dimensional vectors, BGELargeZHV15 uses 1024-dimensional vectors. Example for BGESmallZHV15:

```bash
curl -X PUT "http://localhost:6333/collections/cases" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 512,
      "distance": "Cosine"
    }
  }'
```

3. Configure `config.toml` with your settings.
4. Run the application:

```bash
cargo build --release
```

or with CUDA support so that it can use GPU :
  - Download onnxruntime with CUDA support from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime/releases)
  - Unzip the package and set the environment variable `ORT_DYLIB_PATH` to point to the `libonnxruntime.so` file in the unzipped folder.
  - Build with the `cuda` feature: `cargo build --release -F cuda`
  - Run with CUDA support:
    `ORT_DYLIB_PATH=/path/to/libonnxruntime.so target/release/vsearch` [Documentation](https://ort.pyke.io/setup/linking#dynamic-linking)

Note: Ensure you could connect to the Hugging Face, you may need to set the `HTTPS_PROXY` and `HTTP_PROXY` environment variables.