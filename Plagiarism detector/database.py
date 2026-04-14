import json
import os
from endee import Endee, Precision

# ─── Config ────────────────────────────────────────────────────────────────────
ENDEE_HOST = os.getenv("ENDEE_HOST", "http://localhost:8080/api/v1")
INDEX_NAME  = "plagiarism_index"
DIMENSION   = 384   # all-MiniLM-L6-v2 output dimension

# Local JSON store to map IDs → titles & content (since Endee meta is lightweight)
STORE_FILE = "doc_store.json"


# ─── Helpers ───────────────────────────────────────────────────────────────────
def _load_store() -> dict:
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_store(store: dict):
    with open(STORE_FILE, "w") as f:
        json.dump(store, f, indent=2)

def _doc_id(title: str) -> str:
    """Create a safe ID from a document title."""
    return title.strip().lower().replace(" ", "_")[:48]


# ─── Endee Client ──────────────────────────────────────────────────────────────
def _get_client() -> Endee:
    client = Endee()
    client.set_base_url(ENDEE_HOST)
    return client


# ─── Index Management ──────────────────────────────────────────────────────────
def init_index():
    """
    Connect to Endee and create the plagiarism index if it doesn't exist.
    Returns the index object ready for upsert/query operations.
    """
    client = _get_client()

    # Check if index already exists
    existing = [idx["name"] for idx in client.list_indexes()]
    if INDEX_NAME not in existing:
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",      # cosine similarity = best for text
            precision=Precision.INT8  # balanced speed + accuracy
        )

    return client.get_index(name=INDEX_NAME)


# ─── Store Document ────────────────────────────────────────────────────────────
def store_document(index, title: str, content: str, embedding: list[float]):
    """
    Store a document's embedding in Endee and save metadata locally.
    """
    doc_id = _doc_id(title)

    # Upsert into Endee
    index.upsert([
        {
            "id": doc_id,
            "vector": embedding,
            "meta": {"title": title},
            "filter": {"type": "document"}
        }
    ])

    # Save full content locally (Endee meta has size limits)
    store = _load_store()
    store[doc_id] = {"title": title, "content": content}
    _save_store(store)


# ─── Retrieve Documents ────────────────────────────────────────────────────────
def get_all_documents() -> list[str]:
    """Return list of all stored document titles."""
    store = _load_store()
    return [v["title"] for v in store.values()]

def get_document_content(doc_id: str) -> dict:
    """Return title and content for a given doc_id."""
    store = _load_store()
    return store.get(doc_id, {})
