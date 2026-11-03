import numpy as np
from src.embeddings import embed_query, embed_texts
from src.graph_db import run_query


def vector_search(question, k=5, pool=30, min_local=2, local_floor=0.55):
    emb = embed_query(question)
    rows = run_query(
        """
        CALL db.index.vector.queryNodes('chunk_embedding', $pool, $emb)
        YIELD node, score
        MATCH (d:Document)-[:HAS_CHUNK]->(node)
        RETURN node.id AS id, node.text AS text, score, d.title AS title, d.url AS url
        ORDER BY score DESC
        """,
        {"pool": pool, "emb": emb},
    )

    local = [r for r in rows if r["url"].startswith("local/") and r["score"] >= local_floor]
    picked = local[:min_local]
    picked_ids = {r["id"] for r in picked}
    picked += [r for r in rows if r["id"] not in picked_ids][: k - len(picked)]
    return sorted(picked, key=lambda r: r["score"], reverse=True)


def seed_entities(chunk_ids, limit=8):
    # only entities mentioned in at least 2 of the top chunks
    return run_query(
        """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE c.id IN $ids
        WITH e, count(c) AS freq
        WHERE freq >= 2
        RETURN e.key AS key, e.name AS name, freq
        ORDER BY freq DESC
        LIMIT $limit
        """,
        {"ids": chunk_ids, "limit": limit},
    )


def drop_hubs(keys, max_degree=20):
    """Remove very well-connected entities so expansion doesn't drift through them."""
    if not keys:
        return []
    rows = run_query(
        """
        MATCH (e:Entity)
        WHERE e.key IN $keys
        OPTIONAL MATCH (e)-[r]-(:Entity)
        WITH e, count(r) AS deg
        WHERE deg <= $max_degree
        RETURN e.key AS key
        """,
        {"keys": list(keys), "max_degree": max_degree},
    )
    return [r["key"] for r in rows]


def expand(keys, limit=40):
    # specific relations first, vague RELATED_TO last
    return run_query(
        """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.key IN $keys OR b.key IN $keys
        WITH a, b, type(r) AS relation
        RETURN a.key AS source_key, a.name AS source, relation,
               b.key AS target_key, b.name AS target
        ORDER BY CASE WHEN relation = 'RELATED_TO' THEN 1 ELSE 0 END
        LIMIT $limit
        """,
        {"keys": keys, "limit": limit},
    )


def rank_facts(question, facts, top_n=15):
    """Keep the facts most semantically similar to the question."""
    if not facts:
        return []
    q = embed_query(question)
    texts = [f"{s} {r.replace('_', ' ').lower()} {t}" for s, r, t in facts]
    scores = [float(np.dot(q, v)) for v in embed_texts(texts)]
    ranked = sorted(zip(scores, facts), key=lambda x: x[0], reverse=True)
    return [fact for _, fact in ranked[:top_n]]


def retrieve(question, k=5, hops=2, pool=80, top_facts=15):
    chunks = vector_search(question, k)
    url_order = {}
    for c in chunks:
        c["src_n"] = url_order.setdefault(c["url"], len(url_order) + 1)

    seeds = seed_entities([c["id"] for c in chunks])

    frontier = [s["key"] for s in seeds]
    seen_keys = set(frontier)
    facts, seen_facts = [], set()

    for _ in range(hops):
        if not frontier or len(facts) >= pool:
            break
        next_frontier = set()
        for row in expand(frontier, pool):
            fact = (row["source"], row["relation"], row["target"])
            if fact not in seen_facts:
                seen_facts.add(fact)
                facts.append(fact)
            for key in (row["source_key"], row["target_key"]):
                if key not in seen_keys:
                    seen_keys.add(key)
                    next_frontier.add(key)
        frontier = drop_hubs(next_frontier)

    return {
        "chunks": chunks,
        "seeds": [s["name"] for s in seeds],
        "facts": rank_facts(question, facts, top_facts),
    }

def format_context(result):
    facts = "\n".join(f"- {s} {r} {t}" for s, r, t in result["facts"])
    passages = "\n\n".join(
        f"[{c['src_n']}] ({'PAPER' if c['url'].startswith('local/') else 'WEB'}: {c['title']}) {c['text']}"
        for c in result["chunks"]
    )
    return f"GRAPH FACTS:\n{facts}\n\nPASSAGES:\n{passages}"


if __name__ == "__main__":
    q = "How is QLoRA related to LoRA and parameter-efficient fine-tuning?"
    result = retrieve(q)

    print("QUESTION:", q)
    print("\nTOP CHUNKS:")
    for c in result["chunks"]:
        print(f"  {c['score']:.3f}  {c['title'][:60]}")
    print("\nSEED ENTITIES:", result["seeds"])
    print("\nGRAPH FACTS:")
    for s, r, t in result["facts"]:
        print(f"  {s} -{r}-> {t}")