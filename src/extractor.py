import json
from groq import Groq
from pydantic import BaseModel
from config import GROQ_API_KEY, EXTRACTION_MODEL, ENTITY_TYPES, RELATION_TYPES

_client = Groq(api_key=GROQ_API_KEY)


class Entity(BaseModel):
    name: str
    type: str


class Relationship(BaseModel):
    source: str
    relation: str
    target: str


class Extraction(BaseModel):
    entities: list[Entity] = []
    relationships: list[Relationship] = []


SYSTEM_PROMPT = f"""You extract a knowledge graph about artificial intelligence from text.

Reply with JSON only, in exactly this shape:
{{"entities": [{{"name": "...", "type": "..."}}],
  "relationships": [{{"source": "...", "relation": "...", "target": "..."}}]}}

Rules:
- Entity "type" must be one of: {", ".join(ENTITY_TYPES)}
- Relationship "relation" must be one of: {", ".join(RELATION_TYPES)}
- Direction matters. Read each relation as "source RELATION target":
  LoRA INTRODUCED_BY Microsoft, QLoRA BASED_ON LoRA, OLoRA BASED_ON LoRA,
  LoRA IS_A Parameter-Efficient Fine-Tuning, LoRA APPLIED_TO GPT-3,
  Hugging Face PEFT IMPLEMENTS LoRA, TRL USES LoRA,
  Model EVALUATED_ON Dataset, ModelA OUTPERFORMS ModelB.
- USES: a tool or system uses a method or component (TRL USES LoRA).
  Never say a method uses a tool.
- Names: write the technique as "Parameter-Efficient Fine-Tuning", never bare "PEFT".
  Write the library as "Hugging Face PEFT". Expand other acronyms the same way when the text allows.
- Paper means an academic research paper, named by its full title. Never name a paper
  after a method (no "LoRA (paper)"). If the title is not in the text, skip it.
- APPLIED_TO targets must be specific named models or datasets, not broad fields
  like NLP, Vision or Speech Recognition.
- Skip generic terms such as "Base Model", "Linear Algebra" or "Float16".
- Prefer the specific relations. Use RELATED_TO only when nothing else fits.
- Ignore code identifiers (function, class and parameter names), ads, navigation,
  sidebars, and any companies or reports the page merely publishes, sells or promotes.
- Only extract what the text states. Do not invent anything.
- Every relationship source and target must also appear in "entities".
- If the text has nothing relevant to AI, return empty lists."""


def _clean(ex):
    """Enforce the schema: drop unknown types/relations and dangling relationships."""
    type_map = {t.lower(): t for t in ENTITY_TYPES}

    entities, names = [], set()
    for e in ex.entities:
        etype = type_map.get(e.type.strip().lower())
        name = e.name.strip()
        if not etype or not name:
            continue
        entities.append(Entity(name=name, type=etype))
        names.add(name.lower())

    relationships = []
    for r in ex.relationships:
        rel = r.relation.strip().upper().replace(" ", "_")
        src, tgt = r.source.strip(), r.target.strip()
        if rel not in RELATION_TYPES:
            continue
        if src.lower() in names and tgt.lower() in names and src.lower() != tgt.lower():
            relationships.append(Relationship(source=src, relation=rel, target=tgt))

    return Extraction(entities=entities, relationships=relationships)


class RateLimited(Exception):
    pass


def extract_from_chunk(text, retries=2):
    """Extract entities and relationships from one chunk of text."""
    for attempt in range(retries + 1):
        try:
            resp = _client.chat.completions.create(
                model=EXTRACTION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                reasoning_effort="low",
            )
            data = json.loads(resp.choices[0].message.content)
            return _clean(Extraction(**data))
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                raise RateLimited(str(e))
            if attempt == retries:
                print(f"extraction failed: {e}")
    return Extraction()

if __name__ == "__main__":
    sample = (
        "LoRA (Low-Rank Adaptation) was introduced by Microsoft researchers. "
        "It is a parameter-efficient fine-tuning method based on low-rank matrix "
        "decomposition and is widely used to adapt models like LLaMA. "
        "The Hugging Face PEFT library implements LoRA. QLoRA builds on LoRA and "
        "was evaluated on the MMLU benchmark."
    )
    result = extract_from_chunk(sample)
    print("ENTITIES")
    for e in result.entities:
        print(f"  {e.name} [{e.type}]")
    print("RELATIONSHIPS")
    for r in result.relationships:
        print(f"  {r.source} -{r.relation}-> {r.target}")