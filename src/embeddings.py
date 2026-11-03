from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    vectors = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text):
    return embed_texts([text])[0]


if __name__ == "__main__":
    import numpy as np

    q, a, b = embed_texts([
        "What is LoRA?",
        "LoRA is a parameter-efficient fine-tuning method.",
        "The best way to cook pasta is in salted water.",
    ])
    print("dimension:", len(q))
    print("similarity (LoRA question vs LoRA answer):", round(float(np.dot(q, a)), 3))
    print("similarity (LoRA question vs pasta):", round(float(np.dot(q, b)), 3))