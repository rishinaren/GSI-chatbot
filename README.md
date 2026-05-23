# GSI Standards RAG

Citation-first RAG chatbot for geotextiles / standards research.

## Local development

```bash
pip install -e ".[api,pdf,pinecone,llm,auth,aws,dev]"
standards-rag ingest documents/ --out data/index/standards-index.json
uvicorn standards_rag.api:app --reload
cd frontend && npm install && npm run dev
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for AWS deployment.
