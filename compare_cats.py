import itertools
import numpy as np
import matplotlib.pyplot as plt
from nlputils import simcoefs
from database.make_patent_db import load_session, Patent
from nlputils.preprocessing import FeatureTransform

session = load_session()

# load all patents from subcategories A61M (10.881) and A61F (14.750)
patsf = session.query(Patent).filter(Patent.category.contains('A61F'))
patsm = session.query(Patent).filter(Patent.category.contains('A61M'))

# make dict {patent_id: patent_category}
patsf_dict = {pat.id : pat.category for pat in patsf}
patsm_dict = {pat.id : pat.category for pat in patsm}

def make_corpus(patsm, patsf):
    # make a corpus with the text of all patents: random, cited, duplicate and target docs
    corpus = {}
    print 'going through A61F'
    for pat in patsf:
        corpus[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8') + '\n'
                          + pat.description.encode('utf-8') + '\n' + pat.claims.encode('utf-8'))
    print 'going through A61M'
    for pat in patsm:
        corpus[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8') + '\n'
                          + pat.description.encode('utf-8') + '\n' + pat.claims.encode('utf-8'))
    return corpus

# once performed and saved, no need to do it again
'''
corpus = make_corpus(patsm, patsf)
np.save('db_statistics/corpus.npy', corpus)
patsm_sample = np.random.choice(patsm.count(), 1000, replace=False)
patsf_sample = np.random.choice(patsf.count(), 1000, replace=False)
patsm_ids = [patsm_dict.keys()[i] for i in patsm_sample]
patsf_ids = [patsf_dict.keys()[i] for i in patsf_sample]
np.save('db_statistics/patsm_ids.npy', patsm_ids)
np.save('db_statistics/patsf_ids.npy', patsf_ids)
'''
# load from disk instead
corpus = np.load('db_statistics/corpus.npy').item()
patsf_ids = np.load('db_statistics/patsf_ids.npy')
patsm_ids = np.load('db_statistics/patsm_ids.npy')

# extract features for the corpus of all patents
ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
pat_feats = ft.texts2features(corpus)

# make lists to store sim scores
within_m = {}
within_f = {}
between_mf = {}
#Calculate the sim scores
for simcoef in ['linear', 'jaccard']:
    print "Evaluating for %s" %simcoef
    # compare all patents from category A61M with each other
    for combi in itertools.combinations(patsm_ids, 2):
        within_m[combi] = simcoefs.compute_sim(pat_feats[combi[0]], pat_feats[combi[1]], simcoef)
    # same for category A61F
    for combi in itertools.combinations(patsf_ids, 2):
        within_f[combi] = simcoefs.compute_sim(pat_feats[combi[0]], pat_feats[combi[1]], simcoef)
    # compare all patent pairs (i, j) where i is from category A61M and j from A61F
    for combi in itertools.product(patsm_ids, patsf_ids):
        between_mf[combi] = simcoefs.compute_sim(pat_feats[combi[0]], pat_feats[combi[1]], simcoef)
    fig = plt.figure()

    plt.hist(within_m.values(), bins=100, color='r', normed=True, histtype='step', label='A61M')
    plt.hist(within_f.values(), bins=100, color='b', normed=True, histtype='step', label='A61F')
    plt.hist(between_mf.values(), bins=100, color='g', normed=True, histtype='step', label='A61M~A61F')
    plt.legend()
    plt.savefig('db_statistics/cat_distributions_%s.pdf' %simcoef)
    plt.clf()