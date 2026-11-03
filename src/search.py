from tavily import TavilyClient
from config import TAVILY_API_KEY

MAX_CHARS = 8000  # cap per page to keep LLM token use under control

_client = TavilyClient(api_key=TAVILY_API_KEY)


def search_topic(topic, max_results=5):
    """Search the web for an AI topic and return a list of documents."""
    response = _client.search(
        query=topic,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=True,
    )

    docs = []
    for r in response["results"]:
        text = r.get("raw_content") or r.get("content") or ""
        text = text.strip()
        if len(text) < 200:  # skip pages with too little content
            continue
        docs.append({
            "url": r["url"],
            "title": r.get("title", ""),
            "text": text[:MAX_CHARS],
        })
    return docs


if __name__ == "__main__":
    docs = search_topic("LoRA parameter-efficient fine-tuning", max_results=3)
    for d in docs:
        print(d["title"])
        print(d["url"])
        print(len(d["text"]), "chars")
        print("---")