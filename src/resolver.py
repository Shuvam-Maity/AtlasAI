import re

# keys that must NOT have a trailing "s" stripped
PROTECT = {"transformers", "rags"}

# normalized alias -> normalized canonical key (extend as you spot duplicates)
ALIASES = {
    "lowrankadaptation": "lora",
    "retrievalaugmentedgeneration": "rag",
    "largelanguagemodel": "llm",
    "parameterefficientfinetuning": "peft",
    "stanforduniversity": "stanford",
    "pagerankalgorithm": "pagerank",
    "duallevelretrievalsystem": "duallevelretrieval",
    "duallevelretrievalparadigm": "duallevelretrieval",
    "communitylevelsummary": "communitysummary",
    "maskedlanguagemodeling": "maskedlanguagemodel",
    "promptengineeringtechnique": "promptengineering",
    "adaptivebenchmarkingprocedure": "adaptivebenchmarking",
}


def _singular(key):
    if key in PROTECT:
        return key
    if key.endswith("ies") and len(key) > 4:
        return key[:-3] + "y"
    if key.endswith("s") and not key.endswith("ss") and len(key) > 3:
        return key[:-1]
    return key


def normalize_key(name):
    """Lowercase, drop punctuation, 'v2.0' -> '2.0', and singularize."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    key = re.sub(r"v(?=\d)", "", key)
    key = _singular(key)
    return ALIASES.get(key, key)


def resolve_extraction(ex):
    """Merge duplicate entities in one extraction and re-point relationships to keys."""
    entities = {}
    for e in ex.entities:
        key = normalize_key(e.name)
        if key and key not in entities:
            entities[key] = {"key": key, "name": e.name, "type": e.type}

    relationships, seen = [], set()
    for r in ex.relationships:
        s, t = normalize_key(r.source), normalize_key(r.target)
        if s in entities and t in entities and s != t and (s, r.relation, t) not in seen:
            seen.add((s, r.relation, t))
            relationships.append({"source": s, "relation": r.relation, "target": t})

    return {"entities": list(entities.values()), "relationships": relationships}