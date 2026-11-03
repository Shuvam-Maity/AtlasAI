import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXTRACTION_MODEL = "openai/gpt-oss-120b"
GUARD_MODEL = "openai/gpt-oss-20b"
ANSWER_MODEL = "openai/gpt-oss-120b"

REQUIRED = {
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USERNAME": NEO4J_USERNAME,
    "NEO4J_PASSWORD": NEO4J_PASSWORD,
    "TAVILY_API_KEY": TAVILY_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
}

ENTITY_TYPES = ["Model", "Method", "Dataset", "Paper", "Organization", "Concept", "Tool"]

RELATION_TYPES = [
    "INTRODUCED_BY",
    "BASED_ON",
    "EVALUATED_ON",
    "USES",
    "APPLIED_TO",
    "IS_A",
    "IMPLEMENTS",
    "OUTPERFORMS",
    "RELATED_TO",
]

if __name__ == "__main__":
    for name, value in REQUIRED.items():
        print(f"{name}: {'set' if value else 'MISSING'}")