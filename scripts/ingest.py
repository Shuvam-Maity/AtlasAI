import sys
import time

from src.search import search_topic
from src.chunking import make_chunks
from src.extractor import extract_from_chunk
from src.resolver import resolve_extraction
from src.loader import load_document
from src.graph_db import setup_constraints, setup_vector_index, run_query, close

MAX_CHUNKS_PER_DOC = 8   # keep the first runs small; raise later
PAUSE_SECONDS = 1.0      # gentle pacing for the LLM API


def ingest_topic(topic, max_results=3):
    print(f"\n=== Topic: {topic} ===")
    docs = search_topic(topic, max_results=max_results)
    print(f"found {len(docs)} usable pages")

    for doc in docs:
        chunks = make_chunks(doc)[:MAX_CHUNKS_PER_DOC]
        print(f"- {doc['title'][:70]} ({len(chunks)} chunks)")

        resolved_list = []
        for chunk in chunks:
            extraction = extract_from_chunk(chunk["text"])
            resolved_list.append(resolve_extraction(extraction))
            time.sleep(PAUSE_SECONDS)

        load_document(doc, chunks, resolved_list)


def print_summary():
    print("\n=== Graph summary ===")
    for row in run_query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label"):
        print(row)
    for row in run_query("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel"):
        print(row)


if __name__ == "__main__":
    setup_constraints()
    setup_vector_index()

    topics = sys.argv[1:] or ["LoRA parameter-efficient fine-tuning"]
    for t in topics:
        ingest_topic(t)

    print_summary()
    close()