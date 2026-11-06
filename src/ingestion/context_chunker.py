import re


def clean_pdf_text(text):
    text = re.sub(
        r"(?<=[a-z])-[ \t]*\n[ \t]*(?=[a-z])",
        "",
        text,
    )

    return text.strip()


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text.strip(),
        )
        if sentence.strip()
    ]


def split_blocks(text):
    blocks = re.split(
        r"\n\s*\n",
        text.strip(),
    )

    return [
        block.strip()
        for block in blocks
        if block.strip()
    ]


def split_large_block(block, chunk_size):
    sentences = split_sentences(block)

    if len(sentences) <= 1:
        return [
            block[i:i + chunk_size].strip()
            for i in range(0, len(block), chunk_size)
            if block[i:i + chunk_size].strip()
        ]

    parts = []
    current = []

    for sentence in sentences:
        candidate = " ".join(
            current + [sentence]
        )

        if current and len(candidate) > chunk_size:
            parts.append(
                " ".join(current)
            )
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        parts.append(
            " ".join(current)
        )

    return parts


def get_overlap(text, overlap):
    if overlap <= 0:
        return ""

    sentences = split_sentences(text)

    if not sentences:
        return ""

    selected = []
    length = 0

    for sentence in reversed(sentences):
        selected.insert(0, sentence)
        length += len(sentence)

        if length >= overlap:
            break

    return " ".join(selected)


def chunk_documents_contextual(
    documents,
    chunk_size=1000,
    overlap=150,
):
    chunks = []

    for document in documents:
        text = document.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        if document["source"].lower().endswith(".pdf"):
            text = clean_pdf_text(text)

        blocks = split_blocks(text)

        pieces = []

        for block in blocks:
            if len(block) <= chunk_size:
                pieces.append(block)
            else:
                pieces.extend(
                    split_large_block(
                        block,
                        chunk_size,
                    )
                )

        current = ""
        chunk_id = 0

        for piece in pieces:
            candidate = (
                f"{current}\n\n{piece}".strip()
                if current
                else piece
            )

            if current and len(candidate) > chunk_size:
                chunks.append(
                    {
                        "text": current,
                        "source": document["source"],
                        "document_id": document["document_id"],
                        "page": document["page"],
                        "chunk_id": chunk_id,
                    }
                )

                chunk_id += 1

                previous = get_overlap(
                    current,
                    overlap,
                )

                current = (
                    f"{previous}\n\n{piece}".strip()
                    if previous
                    else piece
                )

            else:
                current = candidate

        if current:
            chunks.append(
                {
                    "text": current,
                    "source": document["source"],
                    "document_id": document["document_id"],
                    "page": document["page"],
                    "chunk_id": chunk_id,
                }
            )

    return chunks