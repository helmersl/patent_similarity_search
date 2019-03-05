# Finding a patent's prior art using text similarity
This repository contains research work on finding prior art for a given patent.
The approach is to find the most similar documents for a given patent application
by comparing them using similarity measures calculated on the documents' full texts. 
For further details on the experiments please refer to the [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0212103):

```
@article{helmers2019automating,
  title={Automating the search for a patent's prior art with a full text similarity search},
  author={Helmers, Lea and Horn, Franziska and Biegler, Franziska and Oppermann, Tim and M{\"u}ller, Klaus-Robert},
  journal={{PLoS ONE}},
  volume={14},
  number={3},
  pages={e0212103},
  year={2019},
  publisher={Public Library of Science}
}
```

All the data sets needed for reproducing the analyses are available at:
`https://figshare.com` and can be downloaded in a compressed format after sign-up

* SQLite database-file: `https://figshare.com/articles/Patent_Database/7264733`
* Patent scoring by expert and corpus subsample: `https://figshare.com/articles/human_eval_tar_gz/7257215`
* Entire corpus: `https://figshare.com/articles/corpus_tar_gz/7257194`

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
