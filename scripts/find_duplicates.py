import numpy as np

from src.graph_db import run_query, close
from src.embeddings import embed_texts

THRESHOLD = 0.85
SHOW = 60

rows = run_query("MATCH (e:Entity) RETURN e.key AS key, e.name AS name, e.type AS type")
print(len(rows), "entities")

vecs = np.array(embed_texts([r["name"] for r in rows]))
sims = vecs @ vecs.T

pairs = []
for i in range(len(rows)):
    for j in range(i + 1, len(rows)):
        if sims[i][j] >= THRESHOLD:
            pairs.append((float(sims[i][j]), rows[i], rows[j]))

pairs.sort(key=lambda p: -p[0])
for s, a, b in pairs[:SHOW]:
    print(f"{s:.2f}  {a['name']} [{a['type']}]  <->  {b['name']} [{b['type']}]")
print(len(pairs), "candidate pairs in total")

close()