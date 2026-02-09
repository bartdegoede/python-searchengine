import logging

from datasets import load_dataset

from load import DATASET, DATASET_CONFIG

logger = logging.getLogger(__name__)


def download_wikipedia():
    """Pre-download the Wikipedia dataset from Hugging Face.

    The datasets library caches the download in ~/.cache/huggingface/,
    so subsequent calls to load_dataset() will use the cached version.
    """
    logger.info("Downloading %s (%s)...", DATASET, DATASET_CONFIG)
    load_dataset(DATASET, DATASET_CONFIG, split="train")
    logger.info("Done! Dataset is cached and ready to use.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_wikipedia()
