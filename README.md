# AtlasAI

An AI-topic question-answering app built on a **Neo4j knowledge graph (GraphRAG)**. It answers questions about AI and AI-related topics only, with cited sources. Anything else is politely refused.

## How it works

```
Tavily web search ─┐
                   ├─> chunk ─> LLM extraction ─> entity resolution ─> Neo4j
PDF papers ────────┘             (Groq)            (rule-based)     (graph + vector index)

Question ─> topic guard ─> vector search (chunks) ─> graph expansion (2 hops) ─> fact ranking ─> cited answer
```

1. **Ingest:** web pages (Tavily) and PDF papers are split into chunks. An LLM extracts entities and relationships constrained to a fixed schema.
2. **Resolve:** duplicate entities (plurals, hyphens, version spellings) are merged with a rule-based key. Version numbers and model sizes are kept apart on purpose.
3. **Store:** entities, relationships, and chunk embeddings live in one Neo4j database, using its built-in vector index.
4. **Guard:** a small LLM check allows only AI-related questions and fails closed.
5. **Retrieve:** vector search finds relevant text (with guaranteed slots for papers), then graph traversal adds connected facts. Hub entities are skipped and facts are ranked by similarity to the question.
6. **Answer:** the LLM answers only from the retrieved context, cites sources, and says when something is missing.

## Graph schema

- Entity types: Model, Method, Dataset, Paper, Organization, Concept, Tool
- Relations: INTRODUCED_BY, BASED_ON, EVALUATED_ON, USES, APPLIED_TO, IS_A, IMPLEMENTS, OUTPERFORMS, RELATED_TO
- Structure: (Document)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity)

## Stack

Python, Neo4j Aura (graph + vector index), Groq (gpt-oss), Tavily, sentence-transformers (all-MiniLM-L6-v2), LangChain text splitters, Streamlit, pyvis.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=
TAVILY_API_KEY=
GROQ_API_KEY=
```

## Usage

```bash
python -m scripts.ingest "retrieval-augmented generation RAG explained"   # web topics
python -m scripts.ingest_docs                                            # PDFs in docs/ (resumable)
python -m scripts.merge_entities --apply                                 # merge duplicates
python -m scripts.ask                                                    # command-line Q&A
streamlit run app.py                                                     # web UI
```

## Current graph

905 entities, 1,163 relationships and 416 text chunks, built from 17 web pages and 10 research papers. Five papers are fully ingested (GraphRAG and LightRAG still in progress): GraphRAG, LightRAG, LoRA, QLoRA and Attention Is All You Need. Five more (BERT, ReAct, Self-RAG, HippoRAG, RAG) are loaded only partially, as their first 25 chunks.

## Limitations

- Extraction runs on Groq's free tier (200k tokens/day), so papers are ingested over several days. Ingestion is resumable.
- Abstract method descriptions are sometimes missed by the extractor, and code identifiers can leak into the graph.
- Entity merging is rule-based, so unusual spelling variants need manual aliases.
- Relation quality depends on the source: tutorials rarely state authors or results, papers do.