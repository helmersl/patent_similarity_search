import numpy as np
import cPickle as pkl
from word2vec_app import train_word2vec, embed_features
from corpus_utils import make_w2v_corpus
from plot_utils import calc_simcoef_distr, calc_auc, plot_score_distr, make_combis
from nlputils.preprocessing import FeatureTransform


#corpus = np.load('../corpus/corpus.npy').item()
for embed_dim in [200]:
	corpus = np.load('../corpus/corpus.npy').item()
	#pat_corpus = make_w2v_corpus()
	#model = train_word2vec(pat_corpus, 'full_patent_corpus', seed=1, embed_dim=200)
	model = pkl.load(open('human_eval/models/full_patent_corpus_sg_200_hs0_neg13_seed1.model'))
	pat_ids = corpus.keys()
	ft = FeatureTransform(identify_bigrams=False, norm=None, weight='tfidf', renorm='length')
	patfeats = ft.texts2features(corpus)
	featmat_w2v, featurenames = embed_features(model, patfeats, pat_ids)
	patfeats_w2v = {}
	for i, pid in enumerate(pat_ids):
	    patfeats_w2v[pid] = dict(zip(featurenames, featmat_w2v[i,:]))
	#np.save('../corpus/patfeats_w2v.npy', patfeats_w2v)


	#patfeats = np.load('../corpus/patfeats_w2v.npy').item()
	patfeats = patfeats_w2v
	target_ids = np.load('../corpus/target_ids.npy')
	random_ids = np.load('../corpus/random_ids.npy')
	dupl_ids = np.load('../corpus/dupl_ids.npy').item()
	cited_ids = np.load('../corpus/cited_ids.npy').item()

	id_dict = make_combis(target_ids, random_ids, cited_ids, dupl_ids)
	scores = calc_simcoef_distr(patfeats, ['random', 'cited', 'duplicate'], 
	                                      id_dict, 'linear')
	auc = calc_auc(scores['cited'], scores['random'])[2]
	plot_score_distr('human_eval', 'linear', ['random', 'cited'], 
	                {'cited': scores['cited'], 'random': scores['random']},
	                         auc, ['cited'], histdir='word2vec_full%i_nonorm_length' %embed_dim, bins=50)
