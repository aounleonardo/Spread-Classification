# Claim Scraper
## Usage
```
scrapy crawl snopes -o test.json
```

<!-- TODO -->
## Dependencies
- Add twitter-dataops to the PYTHONPATH
- Install spacy:
```
conda install -c conda-forge spacy
pip install -U spacy-lookups-data
python -m spacy download en_core_web_sm
```