import string
import gensim
import datetime
import numpy as np
import pandas as pd
from glob import glob
from database.make_patent_db import load_session, Patent

session = load_session()

def make_d2v_corpus(target_ids):
    corpus = np.load('../corpus/corpus.npy').item()
    train_corpus = [gensim.models.doc2vec.TaggedDocument(pat.lower().split(), [pid]) for pid, pat in corpus.items() if pid not in target_ids]
    target_corpus = {pid: corpus[pid].lower().split() for pid in target_ids}
    return train_corpus, target_corpus

def make_w2v_corpus():
    corpus = np.load('../corpus/corpus.npy').item()
    return [pat.lower().split() for pat in corpus.values()]

class PatentCorpus(object):
    """
    Iterates over patents from the database we have human scores for
    Adapt mode for word2vec or doc2vec
    """
    def __init__(self):
        self.mode = None
        self.dir = 'human_eval/patent_sheets/*'
        # attributes for regression mode
        self.single_pat_corpus = {}
        self.pat_ids = []
        self.combis = []
        self.binary_label_pairs = {}
        self.human_label_pairs = {}

    def __iter__(self):
        # go through csv-file by csv-file (containing scoring)
        for csv_sheet in glob(self.dir):
            # read the csv sheet
            pat_df = pd.read_csv(csv_sheet, delimiter='\t', usecols=['id', 'human', 'cited'])
            csv_sheet = csv_sheet.strip('/home/lea/Documents/master_thesis/patent_search/human_eval/patent_sheets/')
            target_id = csv_sheet.strip('_human_eval.csv')
            if self.mode == 'regression':
                self.pat_ids.append(target_id)
            # drop rows with missing values
            pat_df = pat_df.dropna()
            pat_df = pat_df.reset_index()
            # access database to get the patent text
            pat = session.query(Patent).filter(Patent.id == target_id)[0]
            #pat = pat[0]
            # for word2vec, yield the text of the target patent as a list of words
            if self.mode == 'w2v':
                yield (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8') + '\n'
                                                 + pat.description.encode('utf-8') + '\n' 
                                                 + pat.claims.encode('utf-8')).lower().split() 
            # for doc2vec, yield a TaggedDocument object
            elif self.mode == 'd2v':
                yield gensim.models.doc2vec.TaggedDocument((pat.title.encode('utf-8') + '\n' 
                                            + pat.abstract.encode('utf-8') + '\n'
                                            + pat.description.encode('utf-8') + '\n' 
                                            + pat.claims.encode('utf-8')).translate(string.maketrans("",""), 
                                                                                    string.punctuation).lower().split(), [target_id])
            # for regression, no preprocessing has to be performed
            elif self.mode == 'regression':
                self.single_pat_corpus[target_id] = (pat.title.encode('utf-8') + '\n' 
                                + pat.abstract.encode('utf-8') + '\n'
                                + pat.description.encode('utf-8') + '\n' 
                                + pat.claims.encode('utf-8'))
            elif self.mode == 'wmd':
                self.single_pat_corpus[target_id] = (pat.title.encode('utf-8') + '\n' 
                                                 + pat.abstract.encode('utf-8') + '\n'
                                                 + pat.description.encode('utf-8') + '\n' 
                                                 + pat.claims.encode('utf-8')).lower().split() 
            else:
                print "Mode has to be set to either d2v, w2v, regression or wmd"
            # go through all cited/random patents
            for i in range(len(pat_df)):
                self.pat_ids.append(pat_df.id[i])
                combi = (target_id, pat_df.id[i])
                # save patent pairs
                self.combis.append(combi)
                # save labels
                self.binary_label_pairs[combi] = pat_df.cited[i]
                self.human_label_pairs[combi] = pat_df.human[i]/5.
                pat = session.query(Patent).filter(Patent.id == pat_df.id[i])
                pat = pat[0]
                # yield the text of the cited/random patent
                # as before, for word2vec, yield the text of the target patent as a list of words
                if self.mode == 'w2v':
                    yield (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8') + '\n'
                                                     + pat.description.encode('utf-8') + '\n' 
                                                     + pat.claims.encode('utf-8')).lower().split()
                # for doc2vec, yield a TaggedDocument object
                elif self.mode == 'd2v':
                    yield gensim.models.doc2vec.TaggedDocument((pat.title.encode('utf-8') + '\n' 
                                                + pat.abstract.encode('utf-8') + '\n'
                                                + pat.description.encode('utf-8') + '\n' 
                                                + pat.claims.encode('utf-8')).translate(string.maketrans("",""), 
                                                                                        string.punctuation).lower().split(), [pat_df.id[i]])
                elif self.mode == 'regression':
                    self.single_pat_corpus[pat_df.id[i]] = (pat.title.encode('utf-8') + '\n' 
                                            + pat.abstract.encode('utf-8') + '\n'
                                            + pat.description.encode('utf-8') + '\n' 
                                            + pat.claims.encode('utf-8'))
                elif self.mode == 'wmd':
                    self.single_pat_corpus[pat_df.id[i]] = (pat.title.encode('utf-8') + '\n' 
                                                 + pat.abstract.encode('utf-8') + '\n'
                                                 + pat.description.encode('utf-8') + '\n' 
                                                 + pat.claims.encode('utf-8')).lower().split()
                else:
                    print "Mode has to be set to either d2v, w2v, regression or wmd"