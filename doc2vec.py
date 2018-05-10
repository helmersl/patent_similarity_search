# import modules and set up logging
import gensim
import logging
import numpy as np
import cPickle as pkl
from corpus_utils import PatentCorpus, make_d2v_corpus
from plot_utils import plot_score_distr, calc_simcoef_distr, group_combis, calc_auc, make_combis
from nlputils.dict_utils import norm_dict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_doc2vec(doc_corpus, corpus='patents', size=50):
    # train the model
    model = gensim.models.doc2vec.Doc2Vec(size=size, min_count=5, iter=18, seed=1, workers=1)
    model.build_vocab(doc_corpus)
    model.train(doc_corpus)
    # pickle the entire model to disk, so we can load&resume training later
    saven = "%s_dm_%i_min5_iter18.model" % (corpus, size)
    print "saving model"
    #pkl.dump(model, open("/home/lhelmers/Documents/doc2vec/models/%s"%saven,'wb'), -1)
    pkl.dump(model, open("human_eval/models/%s_no_target_pats"%saven,'wb'), -1)
    return model
    
def make_doc2vec_corpus(model, target_pat_corpus=False):
    patfeats_d2v = {}
    vecsize = len(model.docvecs[0])
    # get doc vecs for training documents
    for pid in model.docvecs.doctags.keys():
        patfeats_d2v[pid] = norm_dict(dict(zip(range(vecsize), model.docvecs[pid])), 'length')
    if target_pat_corpus:
        # infer doc vecs for target patents
        for pid, pat in target_pat_corpus.items():
            patfeats_d2v[pid] = norm_dict(dict(zip(range(vecsize), model.infer_vector(pat))), 'length')
    return patfeats_d2v

def corpus_to_patfeats(model, corpus, target_ids):
    '''
    Transform corpus into doc2vec feature vectors
    Checks if the patents in the corpus are contained in the model
    If so: take the learned document vector
    otherwise: infer the vector
    '''
    patfeats_d2v = {}
    vecsize = len(model.docvecs[0])
    cont = 0
    not_cont = 0
    for pid, pat in corpus.items():
        # check if the patents in the corpus are contained in the model
        if pid in model.docvecs.doctags.keys():
            patfeats_d2v[pid] = norm_dict(dict(zip(range(vecsize), model.docvecs[pid])), 'length')
            cont+=1
        else:
            not_cont+=1
            patfeats_d2v[pid] = norm_dict(dict(zip(range(vecsize), model.infer_vector(pat.lower().split()))), 'length')
    for tid in target_ids:
        patfeats_d2v[tid] = norm_dict(dict(zip(range(vecsize), model.infer_vector(corpus[tid].lower().split()))), 'length')
    print cont, not_cont
    return patfeats_d2v

def infer_patfeats(corpus, model):
    patfeats_d2v = {}
    vecsize = len(model.docvecs[0])
    for pid, pat in corpus.items():
        patfeats_d2v[pid] = norm_dict(dict(zip(range(vecsize), model.infer_vector(pat.lower().split()))), 'length')
    return patfeats_d2v

def apply_d2v_sectionwise():
    '''
    Evaluate doc2vec for comparison of patent sections
    '''
    # define embedding size
    size=50
    # load data
    model = pkl.load(open("../doc2vec/models/full_pat_corpus_dm_50_min5_iter18.model"))
    corpus = np.load('../corpus/corpus_claims.npy').item()
    target_ids = np.load('../corpus/target_ids.npy')
    random_ids = np.load('../corpus/random_ids.npy')
    dupl_ids = np.load('../corpus/dupl_ids.npy').item()
    cited_ids = np.load('../corpus/cited_ids.npy').item()
    id_dict = make_combis(target_ids, random_ids, cited_ids, dupl_ids)
    patfeats_d2v = infer_patfeats(corpus, model)
    scores = calc_simcoef_distr(patfeats_d2v, ['cited', 'duplicate', 'random'], 
                                              id_dict, 'linear')
    auc = calc_auc(scores['cited'], scores['random'])[2]
    '''
    # guarantee that scores range between 0 and 1
    for label, vals in scores.items():
        scores[label] = scores[label] - np.min(scores[label])
        scores[label] = scores[label]/np.max(scores[label])
    '''
    plot_score_distr('human_eval', 'linear', ['random', 'cited', 'duplicate'], 
                    {'cited': scores['cited'], 'random': scores['random'], 'duplicate': scores['duplicate']},
                             auc, ['cited'], histdir='doc2vec_full%i_claims' %size, bins=50)

def apply_d2v_full_corpus():
    target_ids = np.load('../corpus/target_ids.npy')
    random_ids = np.load('../corpus/random_ids.npy')
    dupl_ids = np.load('../corpus/dupl_ids.npy').item()
    cited_ids = np.load('../corpus/cited_ids.npy').item()
    id_dict = make_combis(target_ids, random_ids, cited_ids, dupl_ids)
    pat_corpus = np.load('../corpus/corpus.npy').item()
    for size in [50]:
        pat_corpus, target_pat_corpus = make_d2v_corpus(target_ids)
        #train model
        model = pkl.load(open("../doc2vec/models/full_pat_corpus_dm_50_min5_iter18.model"))
        #load model
        #model = pkl.load(open('human_eval/patents_dm_50_min5_iter10.model'))
        #patfeats_d2v = infer_patfeats(pat_corpus, model)
        #patfeats_d2v = corpus_to_patfeats(model, pat_corpus, target_ids)
        patfeats_d2v = make_doc2vec_corpus(model, target_pat_corpus)
        #np.save('../doc2vec/patfeats_d2v%i.npy' %size, patfeats_d2v)

        scores = calc_simcoef_distr(patfeats_d2v, ['cited', 'duplicate', 'random'], 
                                              id_dict, 'linear')
        auc = calc_auc(scores['cited'], scores['random'])[2]
        '''
        # guarantee that scores range between 0 and 1
        for label, vals in scores.items():
            scores[label] = scores[label] - np.min(scores[label])
            scores[label] = scores[label]/np.max(scores[label])
        '''
        plot_score_distr('human_eval', 'linear', ['random', 'cited', 'duplicate'], 
                        {'cited': scores['cited'], 'random': scores['random'], 'duplicate': scores['duplicate']},
                                 auc, ['cited'], histdir='doc2vec_full%i_no_target' %size, bins=50)

def apply_d2v_rel_corpus():
    """
    Evaluate the doc2vec feature vectors on the smaller corpus for
    cited/random and relevant/irrelevant labellings
    """
    # load text
    #pat_corpus = PatentCorpus()
    #pat_corpus.mode = 'd2v'
    #list(pat_corpus)
    combis = np.load('human_eval/corpus_info/combis.npy')
    target_ids = list(set([comb[0] for comb in combis]))
    #pat_corpus = np.load('human_eval/doc2vec/corpus.npy')
    #pat_corpus = [gensim.models.doc2vec.TaggedDocument(a[0], a[1]) for a in pat_corpus if a[1][0] not in target_ids]
    ## Plot AUC
    # load model trained on entire patent corpus
    model = pkl.load(open("../doc2vec/models/full_pat_corpus_dm_50_min5_iter18.model"))
    #model = pkl.load(open("../doc2vec/models/full_pat_corpus_dm_50_min5_iter18.model"))
    #model = train_doc2vec(pat_corpus)
    # get doc2vec feature vectors
    single_pat_corpus = np.load('human_eval/corpus_info/single_pat_corpus.npy').item()
    patfeats_d2v = infer_patfeats(single_pat_corpus, model)
    #patfeats_d2v = corpus_to_patfeats(model, single_pat_corpus, [])
    #patfeats_d2v = make_doc2vec_corpus(model, single_pat_corpus, target_ids)
    pat_ids = np.load('human_eval/corpus_info/pat_ids.npy')
    binary_label_pairs = np.load('human_eval/corpus_info/binary_label_pairs.npy').item()
    human_label_pairs = np.load('human_eval/corpus_info/human_label_pairs.npy').item()
    binary_sim_combis, binary_diff_combis = group_combis(binary_label_pairs)
    human_sim_combis, human_diff_combis = group_combis(human_label_pairs)
    for simcoef in ['linear']:
        binary_scores = calc_simcoef_distr(patfeats_d2v, ['random', 'cited'], 
                                           {'cited': binary_sim_combis, 'random': binary_diff_combis},
                                           simcoef)
        human_scores = calc_simcoef_distr(patfeats_d2v, ['irrelevant', 'relevant'],
                                          {'relevant': human_sim_combis, 'irrelevant': human_diff_combis},
                                          simcoef)
        binary_auc = calc_auc(binary_scores['cited'], binary_scores['random'])[2]
        human_auc = calc_auc(human_scores['relevant'], human_scores['irrelevant'])[2]
        plot_score_distr('human_eval', simcoef, ['random', 'cited'], 
                         {'cited': binary_scores['cited'], 'random': binary_scores['random']},
                         binary_auc, ['cited'], histdir='doc2vec_full50_rel_corp', bins=20)
        plot_score_distr('human_eval', simcoef, ['irrelevant', 'relevant'], 
                 {'relevant': human_scores['relevant'], 'irrelevant': human_scores['irrelevant']},
                 human_auc, ['relevant'], histdir='doc2vec_full50_rel_corp', bins=20)

if __name__ == "__main__":
    apply_d2v_rel_corpus()



