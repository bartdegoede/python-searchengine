from collections.abc import Generator

from datasets import load_dataset

from search.documents import Abstract

DATASET: str = "wikimedia/wikipedia"
DATASET_CONFIG: str = "20231101.en"


def load_documents() -> tuple[int, Generator[Abstract, None, None]]:
    """Load Wikipedia abstracts from HuggingFace.

    Returns (total, iterator) so callers can create fixed-size structures
    (like a memmap) without materializing all documents into memory.
    The HF Dataset is Arrow-backed and memory-mapped, so iterating over it
    doesn't load the full dataset into RAM.
    """
    ds = load_dataset(DATASET, DATASET_CONFIG, split="train")

    def _generate() -> Generator[Abstract, None, None]:
        for doc_id, row in enumerate(ds):
            title: str = row["title"]
            url: str = row["url"]
            # extract first paragraph as abstract
            text: str = row["text"]
            abstract = text.split("\n\n")[0] if text else ""
            yield Abstract(ID=doc_id, title=title, url=url, abstract=abstract)

    return len(ds), _generate()
