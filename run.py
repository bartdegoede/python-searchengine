import gzip
from lxml import etree
import os.path
import requests
import time

from search.timing import timing
from search.documents import Abstract
from search.index import Index
from search.substring_search import substring_search
from search.regex_search import regex_search


def download_wikipedia_abstracts():
    URL = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz'
    with requests.get(URL, stream=True) as r:
        r.raise_for_status()
        with open('data/enwiki.latest-abstract.xml.gz', 'wb') as f:
            # write every 1mb
            for i, chunk in enumerate(r.iter_content(chunk_size=1024*1024)):
                f.write(chunk)
                if i % 10 == 0:
                    print(f'Downloaded {i} megabytes')


def load_documents():
    start = time.time()
    with gzip.open('data/enwiki.latest-abstract.xml.gz', 'rb') as f:
        doc_id = 0
        for _, element in etree.iterparse(f, events=('end',), tag='doc'):
            title = element.findtext('./title')
            url = element.findtext('./url')
            abstract = element.findtext('./abstract')

            yield Abstract(ID=doc_id, title=title, url=url, abstract=abstract)

            if doc_id % 5000 == 0:
                print(f'Loaded {doc_id} documents')
            doc_id += 1
            element.clear()
    end = time.time()
    print(f'Parsing XML took {end - start} seconds')


@timing
def index_documents(documents, index):
    for i, document in enumerate(documents):
        index.index_document(document)
        if i % 5000 == 0:
            print(f'Indexed {i} documents')
    return index


if __name__ == '__main__':
    # this will only download the xml dump if you don't have a copy already;
    # just delete the file if you want a fresh copy
    if not os.path.exists('data/enwiki.latest-abstract.xml.gz'):
        download_wikipedia_abstracts()

    index = index_documents(load_documents(), Index())
    print(f'Index contains {len(index.documents)} documents')

    substring_search(index.documents, 'python')
    regex_search(index.documents, 'python')
    index.search('Python Programming Language', search_type='AND')
    index.search('Python Programming Language', search_type='OR')
    index.search('Python Programming Language', search_type='AND', score=True)
    index.search('Python Programming Language', search_type='OR', score=True)
