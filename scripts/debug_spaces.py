import unicodedata

from groq import Groq
from config import GROQ_API_KEY, ANSWER_MODEL
from src.retriever import retrieve, format_context
from src.qa import SYSTEM_PROMPT

client = Groq(api_key=GROQ_API_KEY)

question = "How do LangChain and LangGraph relate?"
context = format_context(retrieve(question))

resp = client.chat.completions.create(
    model=ANSWER_MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
    ],
    temperature=0.2,
)
raw = resp.choices[0].message.content

print("RAW (repr, shows hidden characters):")
print(repr(raw))

print("\nNON-ASCII CHARACTERS FOUND:")
for ch in sorted({c for c in raw if ord(c) > 127}):
    print(f"  U+{ord(ch):04X}  {unicodedata.name(ch, 'UNKNOWN')}  ({unicodedata.category(ch)})")