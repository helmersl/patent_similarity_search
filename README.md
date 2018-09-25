# Finding a patent's prior art using text similarity
This repository contains research work on finding prior art for a given patent.
The approach is to find the most similar documents for a given patent application
by comparing them using similarity measures calculated on the documents' full texts.

## Compile dataset and load it into sqlite database
### Crawling patent files from google patents
 * Adapt the seed patents in the main functions in patentcollector.py
```
python patentcollector.py
```
### Create SQLite DB
  * Save your patent files as *.csv* files with following metadata as columns: ['id', 'title', 'category', 'pub_number', 'app_number', 'pub_date', 'abstract',
               'description', 'claims', 'cited_patents', 'pub_dates']
  * Adapt the path in the main function of make_patent_db.py to point to the directory containing your patent files
```
python make_patent_db.py
```
## Exploratory data analysis
### Evaluate Corpus statistics
Check out the category distributions in your corpus
```
python compare_cats.py
```
## Run similarity search
#### The different feature extraction methods:
###### *Bag-of-words* with *tf-idf*
```
python idf_regression.py
```
###### *Kernel-PCA*
```
python kpca.py
```
###### *Latent semantic analysis (LSA)*
```
python lat_sem_ana.py
```
###### Word2vec
```
python word2vec_app.py
```
###### Doc2vec
```
python doc2vec.py
```
