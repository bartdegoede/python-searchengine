# python-searchengine
Simple search engine implementation in Python for illustrative purposes. Expects [`enwiki-latest-abstract1.xml.gz`](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract1.xml.gz) to be downloaded **and extracted** in the source directory.

It contains a bunch of print statements to monitor progress, and a hacky way of timing methods.

# Usage

Run from the command line:

```bash
$ wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract1.xml.gz
$ gunzip enwiki-latest-abstract1.xml.gz
$ pip install -r requirements.txt
$ python search_engine.py
```

Run from interactive console:
```python
Python 3.8.0 (v3.8.0:fa919fdf25, Oct 14 2019, 10:23:27)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.17.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from search_engine import *

In [2]: index = run()
# Loads of stdout about parsing XML, takes ±30 seconds on my laptop
Parsing XML took 32.68613815307617 seconds
# Loads of stdout about indexing data, takes ±30 seconds on my laptop
index_documents took 25.253756046295166 seconds
substring_search took 0.40305519104003906 seconds
regex_search took 2.7608580589294434 seconds
search took 0.050067901611328125 milliseconds
run took 61.108176946640015 seconds

In [3]: index.search('python programming LanGuAGe')
search took 0.03814697265625 milliseconds
Out[3]:
[Abstract(ID=277611, title='Wikipedia: Mod python', abstract='mod_python is an Apache HTTP Server module that integrates the Python programming language with the server. It is intended to provide a Python language binding for the Apache HTTP Server.', url='https://en.wikipedia.org/wiki/Mod_python'),
 Abstract(ID=266532, title='Wikipedia: CDex', abstract='| programming language   = C, C++, Python', url='https://en.wikipedia.org/wiki/CDex'),
 Abstract(ID=315182, title='Wikipedia: PSI (computational chemistry)', abstract='| programming language  = C++, Python', url='https://en.wikipedia.org/wiki/PSI_(computational_chemistry)')]
```
