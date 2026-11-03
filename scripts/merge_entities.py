import sys
from collections import defaultdict

from config import RELATION_TYPES
from src.graph_db import run_query, close
from src.resolver import normalize_key

APPLY = "--apply" in sys.argv


def load_groups():
    rows = run_query(
        """
        MATCH (e:Entity)
        OPTIONAL MATCH (e)<-[m:MENTIONS]-()
        RETURN e.key AS key, e.name AS name, e.type AS type, count(m) AS mentions
        """
    )
    groups = defaultdict(list)
    for r in rows:
        groups[normalize_key(r["name"])].append(r)
    return groups


def merge_group(new_key, members):
    members = sorted(members, key=lambda m: (-m["mentions"], len(m["name"])))
    keep, others = members[0], members[1:]
    ck = keep["key"]

    for d in others:
        dk = d["key"]
        run_query(
            """
            MATCH (ch:Chunk)-[:MENTIONS]->(:Entity {key: $dk})
            MATCH (c:Entity {key: $ck})
            MERGE (ch)-[:MENTIONS]->(c)
            """,
            {"dk": dk, "ck": ck},
        )
        for rel in RELATION_TYPES:  # relation names come from our fixed whitelist
            run_query(
                f"""
                MATCH (:Entity {{key: $dk}})-[x:{rel}]->(t:Entity)
                WHERE t.key <> $ck
                MATCH (c:Entity {{key: $ck}})
                MERGE (c)-[y:{rel}]->(t)
                SET y.chunk_ids = coalesce(y.chunk_ids, [])
                    + [cid IN coalesce(x.chunk_ids, []) WHERE NOT cid IN coalesce(y.chunk_ids, [])]
                """,
                {"dk": dk, "ck": ck},
            )
            run_query(
                f"""
                MATCH (s:Entity)-[x:{rel}]->(:Entity {{key: $dk}})
                WHERE s.key <> $ck
                MATCH (c:Entity {{key: $ck}})
                MERGE (s)-[y:{rel}]->(c)
                SET y.chunk_ids = coalesce(y.chunk_ids, [])
                    + [cid IN coalesce(x.chunk_ids, []) WHERE NOT cid IN coalesce(y.chunk_ids, [])]
                """,
                {"dk": dk, "ck": ck},
            )
        run_query("MATCH (d:Entity {key: $dk}) DETACH DELETE d", {"dk": dk})

    if ck != new_key:
        run_query("MATCH (c:Entity {key: $ck}) SET c.key = $nk", {"ck": ck, "nk": new_key})


if __name__ == "__main__":
    groups = load_groups()
    to_merge = {k: v for k, v in groups.items() if len(v) > 1}
    to_rekey = {k: v[0] for k, v in groups.items() if len(v) == 1 and v[0]["key"] != k}
    removed = sum(len(v) - 1 for v in to_merge.values())

    print(f"{sum(len(v) for v in groups.values())} entities, {len(to_merge)} groups to merge "
          f"({removed} nodes would be removed), {len(to_rekey)} keys to update")
    print()

    for new_key, members in sorted(to_merge.items(), key=lambda kv: -len(kv[1]))[:60]:
        names = ", ".join(f"{m['name']} ({m['mentions']})" for m in members)
        print(f"{new_key}: {names}")

    if not APPLY:
        print("\nDry run only. Re-run with --apply to merge.")
    else:
        for new_key, members in to_merge.items():
            try:
                merge_group(new_key, members)
            except Exception as e:
                print(f"failed on {new_key}: {e}")
        for new_key, m in to_rekey.items():
            try:
                run_query("MATCH (c:Entity {key: $ck}) SET c.key = $nk", {"ck": m["key"], "nk": new_key})
            except Exception as e:
                print(f"re-key failed on {m['key']} -> {new_key}: {e}")
        print("\nMerge complete.")

    close()