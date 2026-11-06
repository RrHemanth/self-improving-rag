def chunk_documents(
    documents: list[dict],
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[dict]:
    """Split documents into overlapping chunks while preserving metadata."""

    chunks = []

    for document in documents:
        text = document["text"]

        if not text:
            continue

        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": document["source"],
                        "document_id": document["document_id"],
                        "page": document["page"],
                        "chunk_id": chunk_id,
                    }
                )

            chunk_id += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks