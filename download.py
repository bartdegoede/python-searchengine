import requests


def download_wikipedia_abstracts():
    URL = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz'
    with requests.get(URL, stream=True) as r:
        r.raise_for_status()
        with open('data/enwiki-latest-abstract.xml.gz', 'wb') as f:
            # write every 1mb
            for i, chunk in enumerate(r.iter_content(chunk_size=1024*1024)):
                f.write(chunk)
                if i % 10 == 0:
                    print(f'Downloaded {i} megabytes', end='\r')

                    
if __name__ == '__main__':
    download_wikipedia_abstracts()
