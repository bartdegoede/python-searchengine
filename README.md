# python-searchengine

Simple search engine implementation in Python for illustrative purposes to go with [this blog post](https://bart.degoe.de/building-a-full-text-search-engine-150-lines-of-code/).

## Requirements

Python 3.7 or greater.

## Usage

Run from the command line (this program will exit afterwards):

```bash
$ pip install -r requirements.txt
$ python run.py
Parsing XML took 701.6900851726532 seconds
index_documents took 701.6901140213013 seconds
Index contains 6274026 documents
search took 0.2980232238769531 milliseconds
search took 0.07561087608337402 seconds
search took 0.1850128173828125 milliseconds
search took 0.2769007682800293 seconds
```

Run from interactive console:
```python
Python 3.8.0 (v3.8.0:fa919fdf25, Oct 14 2019, 10:23:27)
Type 'copyright', 'credits' or 'license' for more information
IPython 7.17.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: run run.py
Parsing XML took 701.6900851726532 seconds
index_documents took 701.6901140213013 seconds
Index contains 6274026 documents
search took 0.2980232238769531 milliseconds
search took 0.07561087608337402 seconds
search took 0.1850128173828125 milliseconds
search took 0.2769007682800293 seconds

In [2]: index.search('python programming LanGuAGe'', rank=True)[:5]
search took 0.4620552062988281 milliseconds
Out[2]:
[(Abstract(ID=1018719, title='Wikipedia: Python syntax and semantics', abstract='The syntax of the Python programming language is the set of rules that defines how a Python program will be written and interpreted (by both the runtime system and by human readers). The Python language has many similarities to Perl, C, and Java.', url='https://en.wikipedia.org/wiki/Python_syntax_and_semantics'),
  25.91623091682463),
 (Abstract(ID=2541116, title='Wikipedia: History of Python', abstract="The programming language Python was conceived in the late 1980s, and its implementation was started in December 1989 by Guido van Rossum at CWI in the Netherlands as a successor to ABC capable of exception handling and interfacing with the Amoeba operating system. Van Rossum is Python's principal author, and his continuing central role in deciding the direction of Python is reflected in the title given to him by the Python community, Benevolent Dictator for Life (BDFL).", url='https://en.wikipedia.org/wiki/History_of_Python'),
  25.474817047863255),
 (Abstract(ID=4723564, title='Wikipedia: Zen of Python', abstract='The Zen of Python is a collection of 19 "guiding principles" for writing computer programs that influence the design of the Python programming language. Software engineer Tim Peters wrote this set of principles and posted it on the Python mailing list in 1999.', url='https://en.wikipedia.org/wiki/Zen_of_Python'),
  23.697300894161387),
 (Abstract(ID=2906166, title='Wikipedia: Core Python Programming', abstract='Core Python Programming is a textbook on the Python programming language, written by Wesley J. Chun.', url='https://en.wikipedia.org/wiki/Core_Python_Programming'),
  21.919784740459527),
 (Abstract(ID=2131166, title='Wikipedia: Comparison of programming languages (object-oriented programming)', abstract='This comparison of programming languages compares how object-oriented programming languages such as C++, Java, Smalltalk, Object Pascal, Perl, Python, and others manipulate data structures.', url='https://en.wikipedia.org/wiki/Comparison_of_programming_languages_(object-oriented_programming)'),
  20.407894768933833)]
```

## Using a smaller data set

The code will download the complete set of 6.7m Wikipedia abstracts. If you want to play around with a smaller data set, just pick one of the numbered gzip files from https://dumps.wikimedia.org/enwiki/latest/ and change the URL in [download.py](https://github.com/bartdegoede/python-searchengine/blob/master/download.py#L5).
