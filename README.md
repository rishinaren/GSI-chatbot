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

## Design guidance

`Designing with Geosynthetics`, 6th Edition, is indexed separately from
standards. Its vectors use the Pinecone `design-guidance` namespace and its
runtime metadata uses `data/index/design-guidance.json`.

```bash
python scripts/ingest_design_guidance.py /path/to/books.zip \
  --s3-uri s3://your-runtime-assets-bucket/
```

The command replaces only the design-guidance namespace, verifies its vector
count, and uploads the separate index plus both source PDFs for citations.
