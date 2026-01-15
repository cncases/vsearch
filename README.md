# vsearch

### qdrant

https://github.com/qdrant/qdrant/blob/master/tools/sync-web-ui.sh

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

```yml
log_level: INFO
storage:
  hnsw_index:
    on_disk: true

telemetry_disabled: true
```