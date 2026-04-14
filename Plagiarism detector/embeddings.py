from sentence_transformers import SentenceTransformer

# Load the model once (cached automatically by Python)
# all-MiniLM-L6-v2 produces 384-dimensional embeddings — fast and accurate
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def get_embedding(text: str) -> list[float]:
    """
    Convert a text string into a 384-dimensional embedding vector.
    This vector captures the semantic meaning of the text.
    """
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
