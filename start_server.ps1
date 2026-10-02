$env:EMBEDDING_PROVIDER="mistral"
$env:EMBEDDING_MODEL="mistral-embed"
$env:EMBEDDING_BASE_URL="https://api.mistral.ai/v1"
$env:MISTRAL_API_KEY="<VOTRE_CLE_MISTRAL>"
$env:QDRANT_URL="http://localhost:6333"
$env:QDRANT_API_KEY="your-qdrant-api-key"
$env:LOG_LEVEL="debug"

node build/index.js
