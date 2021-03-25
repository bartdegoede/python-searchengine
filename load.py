import gzip
from lxml import etree
import time

from search.documents import Abstract

def load_documents():
    start = time.time()
    with gzip.open('data/enwiki-latest-abstract.xml.gz', 'rb') as f:
        doc_id = 0
        for _, element in etree.iterparse(f, events=('end',), tag='doc'):
            title = element.findtext('./title')
            url = element.findtext('./url')
            abstract = element.findtext('./abstract')

            yield Abstract(ID=doc_id, title=title, url=url, abstract=abstract)

            doc_id += 1
            element.clear()
    end = time.time()
    print(f'Parsing XML took {end - start} seconds')
