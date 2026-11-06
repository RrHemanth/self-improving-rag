def hit_rate(results, relevant_pages, k=5):
    retrieved_pages = [
        result["page"]
        for result in results[:k]
    ]

    for page in retrieved_pages:
        if page in relevant_pages:
            return 1

    return 0


def reciprocal_rank(results, relevant_pages):
    for rank, result in enumerate(results, 1):
        if result["page"] in relevant_pages:
            return 1 / rank

    return 0


def recall_at_k(results, relevant_pages, k=5):
    retrieved_pages = set(
        result["page"]
        for result in results[:k]
    )

    relevant_pages = set(relevant_pages)

    found = retrieved_pages.intersection(relevant_pages)

    return len(found) / len(relevant_pages)

def precision_at_k(results, relevant_pages, k=5):
    retrieved_pages = [
        result["page"]
        for result in results[:k]
    ]

    relevant_pages = set(relevant_pages)

    relevant_count = sum(
        1 for page in retrieved_pages
        if page in relevant_pages
    )

    return relevant_count / len(retrieved_pages)