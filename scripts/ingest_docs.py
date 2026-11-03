import time
from pathlib import Path

from pypdf import PdfReader

from src.chunking import make_chunks
from src.extractor import extract_from_chunk, RateLimited
from src.resolver import resolve_extraction
from src.loader import load_document
from src.graph_db import setup_constraints, setup_vector_index, run_query, close

DOCS_DIR = Path("docs")
CHUNKS_PER_RUN = 50     # daily budget: lower it if questions start hitting the limit
BATCH_SIZE = 10         # save progress to Neo4j every N chunks
PAUSE_SECONDS = 1.0
PRIORITY = ["LoRA", "GraphRAG", "LightRAG", "QLoRA", "Attention Is All You Need"]


def read_pdf(path):
    reader = PdfReader(str(path))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    cut = text.rfind("\nReferences")
    if cut > len(text) * 0.6:
        text = text[:cut]
    return text.strip()


def existing_ids(url):
    rows = run_query(
        "MATCH (:Document {url: $u})-[:HAS_CHUNK]->(c:Chunk) RETURN c.id AS id", {"u": url}
    )
    return {r["id"] for r in rows}


def pdf_order(path):
    try:
        return PRIORITY.index(path.stem)
    except ValueError:
        return len(PRIORITY)


def run():
    budget = CHUNKS_PER_RUN
    pdfs = sorted(DOCS_DIR.glob("*.pdf"), key=pdf_order)

    for path in pdfs:
        if budget <= 0:
            print("Daily chunk budget used up. Run again later to continue.")
            return

        text = read_pdf(path)
        if len(text) < 500:
            print(f"- {path.name}: no readable text, skipped")
            continue

        url = f"local/{path.name}"
        doc = {"url": url, "title": path.stem, "text": text}
        chunks = make_chunks(doc)
        have = existing_ids(url)
        todo = [c for c in chunks if c["id"] not in have][:budget]

        if not todo and len(have) >= len(chunks):
            print(f"- {path.name}: complete ({len(chunks)}/{len(chunks)})")
            continue

        print(f"- {path.name}: {len(have)}/{len(chunks)} in graph, extracting {len(todo)} more")

        batch, batch_results, stopped = [], [], False
        for i, chunk in enumerate(todo, start=1):
            try:
                result = resolve_extraction(extract_from_chunk(chunk["text"]))
            except RateLimited:
                stopped = True
                break
            batch.append(chunk)
            batch_results.append(result)
            budget -= 1
            time.sleep(PAUSE_SECONDS)

            if len(batch) == BATCH_SIZE:
                load_document(doc, batch, batch_results)
                batch, batch_results = [], []
                print(f"    saved {i}/{len(todo)}")

        if batch:
            load_document(doc, batch, batch_results)
            print(f"    saved {len(batch)} more")

        if stopped:
            print("Rate limit reached. Progress saved; run again later to continue.")
            return

    print("All papers complete.")


if __name__ == "__main__":
    setup_constraints()
    setup_vector_index()
    run()
    close()