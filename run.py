import logging

from load import load_documents
from search.index import Index
from search.timing import timing

logging.basicConfig(level=logging.INFO)
# set httpx logging to WARNING to reduce noise from API calls
logging.getLogger("httpx").setLevel(logging.WARNING)


@timing
def index_documents(documents, index):
    for document in documents:
        index.index_document(document)
    return index


if __name__ == "__main__":
    index = index_documents(load_documents(), Index())
    print(f"Index contains {len(index.documents)} documents")

    index.search("London Beer Flood", search_type="AND")
    index.search("London Beer Flood", search_type="OR")
    index.search("London Beer Flood", search_type="AND", rank=True)
    index.search("London Beer Flood", search_type="OR", rank=True)
