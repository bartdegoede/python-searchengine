from collections import Counter
from dataclasses import dataclass
from lxml import etree
import math
import time
import re
import string
import Stemmer

STOPWORDS = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I', 'it', 'for',
                 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by',
                 'from'])
PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation))

stemmer = Stemmer.Stemmer('english')

def timing(method):
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()

        execution_time = end - start
        if execution_time < 0.001:
            print(f'{method.__name__} took {execution_time*1000} milliseconds')
        else:
            print(f'{method.__name__} took {execution_time} seconds')

        return result
    return timed

@dataclass
class Abstract:
    """Wikipedia abstract"""
    ID: int
    title: str
    abstract: str
    url: str

    @property
    def fulltext(self):
        return ' '.join([self.title, self.abstract])

    def analyze(self):
        self.term_frequencies = Counter(analyze(self.fulltext))

    def term_frequency(self, term):
        return self.term_frequencies.get(term, 0)

class Index:
    def __init__(self):
        self.index = {}
        self.documents = {}

    def index_document(self, document):
        if document.ID not in self.documents:
            self.documents[document.ID] = document
            document.analyze()

        for token in analyze(document.fulltext):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(document.ID)

    def document_frequency(self, token):
        return len(self.index.get(token, []))

    def inverse_document_frequency(self, token):
        # Manning, Hinrich and Schütze use log10, so we do too, even though it
        # doesn't really matter which log we use anyway
        # https://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html
        return math.log10(len(self.documents) / self.document_frequency(token))

    def _results(self, analyzed_query):
        return [self.index.get(token, set()) for token in analyzed_query]

    @timing
    def search(self, query, search_type='AND', score=False):
        """
        Boolean search; this will return documents that contain words from the
        query, but not rank them (sets are fast, but unordered).
        """
        analyzed_query = analyze(query)
        results = self._results(analyzed_query)
        if search_type == 'AND':
            # all tokens must be in the document
            documents = [self.documents[doc_id] for doc_id in set.intersection(*results)]
        if search_type == 'OR':
            # only one token has to be in the document
            documents = [self.documents[doc_id] for doc_id in set.union(*results)]

        if score:
            return self._score(analyzed_query, documents)
        return documents

    def _score(self, analyzed_query, documents):
        results = []
        if not documents:
            return results
        for document in documents:
            score = 0.0
            for token in analyzed_query:
                tf = document.term_frequency(token)
                idf = self.inverse_document_frequency(token)
                score += tf * idf
            results.append((document, score))
        return sorted(results, key=lambda doc: doc[1], reverse=True)

def load_documents():
    start = time.time()
    with open('enwiki-latest-abstract1.xml', 'rb') as f:
        ID = 0
        for _, element in etree.iterparse(f, events=('end',), tag='doc'):
            title = element.findtext('./title')
            url = element.findtext('./url')
            abstract = element.findtext('./abstract')

            yield Abstract(ID=ID, title=title, url=url, abstract=abstract)

            if ID % 2500 == 0:
                print(f'Loaded {ID} documents')
            ID += 1
            element.clear()
    end = time.time()
    print(f'Parsing XML took {end - start} seconds')

@timing
def substring_search(documents, query):
    results = []
    for document in documents:
        if query in document.fulltext:
            results.append(document)
    return results

@timing
def regex_search(documents, query):
    results = []
    for document in documents:
        if re.search(rf'\b{query}\b', document.fulltext, re.IGNORECASE):
            results.append(document)
    return results

def tokenize(text):
    return text.split()

def lowercase_filter(tokens):
    return [token.lower() for token in tokens]

def punctuation_filter(tokens):
    return [PUNCTUATION.sub('', token) for token in tokens]

def stopword_filter(tokens):
    return [token for token in tokens if token not in STOPWORDS]

def stem_filter(tokens):
    return stemmer.stemWords(tokens)

def analyze(text):
    tokens = tokenize(text)
    tokens = lowercase_filter(tokens)
    tokens = punctuation_filter(tokens)
    tokens = stopword_filter(tokens)
    tokens = stem_filter(tokens)

    return [token for token in tokens if token]

@timing
def index_documents(documents, index):
    for i, document in enumerate(documents):
        index.index_document(document)
        if i % 2500 == 0:
            print(f'Indexed {i} documents')
    return index


@timing
def run():
    documents = [document for document in load_documents()]
    index = index_documents(documents, Index())

    substring_search(documents, 'python')
    regex_search(documents, 'python')
    index.search('Python Programming Language', search_type='AND')
    index.search('Python Programming Language', search_type='OR')
    index.search('Python Programming Language', search_type='AND', score=True)
    index.search('Python Programming Language', search_type='OR', score=True)
    return index

if __name__ == '__main__':
    run()
