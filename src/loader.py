from config import RELATION_TYPES
from src.graph_db import run_query, setup_constraints, setup_vector_index
from src.embeddings import embed_texts


def load_document(doc, chunks, resolved_list):
    """
    doc:           {"url", "title", "text"}
    chunks:        list of {"id", "text", ...}
    resolved_list: one resolver output per chunk, same order as chunks
    """
    embeddings = embed_texts([c["text"] for c in chunks])

    run_query(
        "MERGE (d:Document {url: $url}) SET d.title = $title",
        {"url": doc["url"], "title": doc["title"]},
    )

    for chunk, emb, resolved in zip(chunks, embeddings, resolved_list):
        run_query(
            """
            MATCH (d:Document {url: $url})
            MERGE (c:Chunk {id: $id})
            SET c.text = $text, c.embedding = $embedding
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            {"url": doc["url"], "id": chunk["id"], "text": chunk["text"], "embedding": emb},
        )

        if resolved["entities"]:
            run_query(
                """
                MATCH (c:Chunk {id: $id})
                UNWIND $entities AS e
                MERGE (n:Entity {key: e.key})
                ON CREATE SET n.name = e.name, n.type = e.type
                MERGE (c)-[:MENTIONS]->(n)
                """,
                {"id": chunk["id"], "entities": resolved["entities"]},
            )

        for rel in RELATION_TYPES:  # relation names come from our fixed whitelist
            rels = [r for r in resolved["relationships"] if r["relation"] == rel]
            if not rels:
                continue
            run_query(
                f"""
                UNWIND $rels AS r
                MATCH (a:Entity {{key: r.source}})
                MATCH (b:Entity {{key: r.target}})
                MERGE (a)-[x:{rel}]->(b)
                SET x.chunk_ids = CASE
                    WHEN $cid IN coalesce(x.chunk_ids, []) THEN x.chunk_ids
                    ELSE coalesce(x.chunk_ids, []) + $cid
                END
                """,
                {"rels": rels, "cid": chunk["id"]},
            )


if __name__ == "__main__":
    setup_constraints()
    setup_vector_index()

    doc = {"url": "https://example.com/test", "title": "Test doc", "text": ""}
    chunks = [
        {"id": "test_0", "text": "LoRA was introduced by Microsoft and is based on low-rank decomposition."},
        {"id": "test_1", "text": "QLoRA builds on LoRA and was evaluated on MMLU."},
    ]
    resolved_list = [
        {
            "entities": [
                {"key": "lora", "name": "LoRA", "type": "Method"},
                {"key": "microsoft", "name": "Microsoft", "type": "Organization"},
            ],
            "relationships": [{"source": "lora", "relation": "INTRODUCED_BY", "target": "microsoft"}],
        },
        {
            "entities": [
                {"key": "qlora", "name": "QLoRA", "type": "Method"},
                {"key": "lora", "name": "LoRA", "type": "Method"},
                {"key": "mmlu", "name": "MMLU", "type": "Dataset"},
            ],
            "relationships": [
                {"source": "qlora", "relation": "BASED_ON", "target": "lora"},
                {"source": "qlora", "relation": "EVALUATED_ON", "target": "mmlu"},
            ],
        },
    ]

    load_document(doc, chunks, resolved_list)

    print(run_query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label"))
    print(run_query("MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel"))