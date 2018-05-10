from corpus_utils import PatentCorpus
import numpy as np
from nlputils.preprocessing import FeatureTransform
from database.make_patent_db import load_session, Patent, Citation
session = load_session()
def make_section_patfeats():
    ## Sanity Check: Make the plots calculating the values again
    # get patent corpus
    pat_corpus = PatentCorpus()
    pat_corpus.mode = 'regression'
    # ugly hack to invoke __iter__() function :-( :
    list(pat_corpus)
    # get relevant information
    #pat_ids = pat_corpus.pat_ids
    #combis = pat_corpus.combis
    #binary_label_pairs = pat_corpus.binary_label_pairs
    #human_label_pairs = pat_corpus.human_label_pairs
    single_pat_corpus = pat_corpus.single_pat_corpus
    # make dict of feature vectors
    ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
    patfeats = ft.texts2features(single_pat_corpus)
    np.save('../corpus/patcorpus_claims.npy', single_pat_corpus)
    np.save('../corpus/patfeats_claims.npy', patfeats)

def make_full_section_corpus():
    corpus_abstract = {}
    corpus_claims = {}
    corpus_description = {}
    corpus = np.load('../corpus/corpus.npy').item()
    for id_ in corpus.keys():
        pat = session.query(Patent).filter(Patent.id == id_)[0]
        corpus_abstract[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8'))
        corpus_claims[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.claims.encode('utf-8'))
        corpus_description[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.description.encode('utf-8'))
    np.save('../corpus/corpus_abstract.npy', corpus_abstract)
    np.save('../corpus/corpus_claims.npy', corpus_claims)
    np.save('../corpus/corpus_description.npy', corpus_description)
if __name__ == "__main__":
    make_full_section_corpus()

