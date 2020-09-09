import re

from .timing import timing

@timing
def regex_search(documents, query):
    results = []
    for document in documents:
        if re.search(rf'\b{query}\b', document.fulltext, re.IGNORECASE):
            results.append(document)
    return results
