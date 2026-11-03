import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1500   # characters
OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)


def make_chunks(doc):
    """Turn a document dict (url, title, text) into chunk dicts with stable ids."""
    doc_hash = hashlib.md5(doc["url"].encode("utf-8")).hexdigest()[:10]
    pieces = _splitter.split_text(doc["text"])
    return [
        {
            "id": f"{doc_hash}_{i}",
            "text": piece.strip(),
            "url": doc["url"],
            "title": doc["title"],
        }
        for i, piece in enumerate(pieces)
        if piece.strip()
    ]


if __name__ == "__main__":
    sample_doc = {
        "url": "https://example.com/lora",
        "title": "Sample",
        "text": ("LoRA is a parameter-efficient fine-tuning method. " * 20 + "\n\n") * 6,
    }
    chunks = make_chunks(sample_doc)
    print("chunks:", len(chunks))
    for c in chunks:
        print(c["id"], len(c["text"]))