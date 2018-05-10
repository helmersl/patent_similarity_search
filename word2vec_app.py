# import modules and set up logging
import word2vec
import logging
import numpy as np
import cPickle as pkl
from corpus_utils import PatentCorpus
from plot_utils import plot_score_distr, calc_simcoef_distr, group_combis, calc_auc
from nlputils.preprocessing import features2mat
from nlputils.simcoefs import compute_sim
from sklearn.preprocessing import normalize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_word2vec(pat_corpus, corpus='patents', seed=1, embed_dim=300):
    """Train the word2vec model"""
    # train the skipgram model; default window=5
    model = word2vec.Word2Vec(pat_corpus, mtype='sg', hs=1, neg=13, embed_dim=embed_dim, seed=seed)
    # delete the huge stupid table again
    model.table = None
    # pickle the entire model to disk, so we can load&resume training later
    saven = "%s_sg_%i_hs0_neg13_seed%i.model" % (corpus, embed_dim, seed)
    print "saving model"
    pkl.dump(model, open("human_eval/models/%s"%saven,'wb'), -1)
    return model
    
def embed_features(model, patfeats, pat_ids):
    """
    Combine BOW-features with word2vec embedding

    Input:
        - model: name of the previously trained w2v model
            where: row 10 of model contains word x --> x = model.index2word[10]
                   row i contains embedding for word x --> i = model.vocab[x].index 
        - patfeats: BOW-features calculated during regression
        - pat_ids: IDs of the patents in the corpus

    Returns:
        - featmat_norm: normalized dot product of BOW-features and w2v-embeddings
        - featurenames: the embedded words
    """
    # normalized or not normalized matrix containing w2v embeddings: nwords x 200
    #w2v_embeddings = model.syn0
    w2v_embeddings = model.syn0norm
    # make feature matrix: ndocs x nwords
    featmat, featurenames = features2mat(patfeats, pat_ids, featurenames=model.index2word)
    # make normalized dot product of BOW- and w2v-matrix: ndocs x 200
    featmat_norm = normalize(featmat.dot(w2v_embeddings), axis=1, norm='l2')
    return featmat_norm, featurenames

if __name__ == "__main__":
    # load text
    sentences = PatentCorpus()
    sentences.mode = 'w2v'
    #train_word2vec(sentences)
    #load model
    model = pkl.load(open('human_eval/models/full_patent_corpus_sg_200_hs0_neg13_seed1.model'))
    # load patfeats and ids saved while performing regression
    pat_ids = np.load('human_eval/corpus_info/pat_ids.npy')
    patfeats = np.load('human_eval/corpus_info/patfeats_human_eval.npy').item()
    #patfeats = np.load('human_eval/corpus_info/patfeats_abstract.npy').item()
    combis = np.load('human_eval/corpus_info/combis.npy')
    binary_label_pairs = np.load('human_eval/corpus_info/binary_label_pairs.npy').item()
    human_label_pairs = np.load('human_eval/corpus_info/human_label_pairs.npy').item()
    # make word2vec embedded feature matrix
    featmat_w2v, featurenames = embed_features(model, patfeats, pat_ids)
    #transform feature matrix into dict
    patfeats_w2v = {}
    for i, pid in enumerate(pat_ids):
        patfeats_w2v[pid] = dict(zip(featurenames, featmat_w2v[i,:]))
    ## Plot AUC

    binary_sim_combis, binary_diff_combis = group_combis(binary_label_pairs)
    human_sim_combis, human_diff_combis = group_combis(human_label_pairs)
    for simcoef in ['linear', 'jaccard']:
        binary_scores = calc_simcoef_distr(patfeats_w2v, ['random', 'cited'], 
                                           {'cited': binary_sim_combis, 'random': binary_diff_combis},
                                           simcoef)
        human_scores = calc_simcoef_distr(patfeats_w2v, ['irrelevant', 'relevant'],
                                          {'relevant': human_sim_combis, 'irrelevant': human_diff_combis},
                                          simcoef)
        binary_auc = calc_auc(binary_scores['cited'], binary_scores['random'])[2]
        human_auc = calc_auc(human_scores['relevant'], human_scores['irrelevant'])[2]
        
        plot_score_distr('human_eval', simcoef, ['random', 'cited'], 
                         {'cited': binary_scores['cited'], 'random': binary_scores['random']},
                         binary_auc, ['cited'], histdir='w2v_sg_200', bins=10)

        plot_score_distr('human_eval', simcoef, ['irrelevant', 'relevant'], 
                 {'relevant': human_scores['relevant'], 'irrelevant': human_scores['irrelevant']},
                 human_auc, ['relevant'], histdir='w2v_sg_200', bins=10)
        
