from embeddings import get_embedding
from database import get_document_content


def check_plagiarism(index, text: str, threshold: float = 0.70) -> list[dict]:
    """
    Check a given text against all stored documents in Endee.

    Args:
        index     : Endee index object
        text      : Text to check for plagiarism
        threshold : Similarity score above which content is flagged (0.0 – 1.0)

    Returns:
        List of matches sorted by similarity (highest first).
        Each item: { title, content, similarity }
    """
    # Step 1: Convert the query text to an embedding
    query_vector = get_embedding(text)

    # Step 2: Search Endee vector database for similar documents
    results = index.query(
        vector=query_vector,
        top_k=5,          # return top 5 most similar documents
        ef=128,           # higher ef = more accurate search
        include_vectors=False
    )

    # Step 3: Build result list with metadata
    matches = []
    for item in results:
        doc_id    = item["id"]
        similarity = item["similarity"]

        # Fetch stored content from local store
        doc_data = get_document_content(doc_id)
        if not doc_data:
            continue

        matches.append({
            "title":      doc_data.get("title", doc_id),
            "content":    doc_data.get("content", ""),
            "similarity": similarity,
            "flagged":    similarity >= threshold
        })

    # Sort by similarity descending
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches
