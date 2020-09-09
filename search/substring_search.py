from .timing import timing

@timing
def substring_search(documents, query):
    results = []
    for document in documents:
        if query in document.fulltext:
            results.append(document)
    return results
