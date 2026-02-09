from datasets import load_dataset
from tqdm import tqdm

from search.documents import Abstract

DATASET = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"


def load_documents():
    ds = load_dataset(DATASET, DATASET_CONFIG, split="train")
    for doc_id, row in enumerate(tqdm(ds, desc="Loading documents")):
        title = row["title"]
        url = row["url"]
        # extract first paragraph as abstract
        text = row["text"]
        abstract = text.split("\n\n")[0] if text else ""

        yield Abstract(ID=doc_id, title=title, url=url, abstract=abstract)
