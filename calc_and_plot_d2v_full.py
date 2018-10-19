import numpy as np
from plot_utils import calc_simcoef_distr, calc_auc, plot_score_distr, make_combis

patfeats = np.load('/Users/leahelmers/Documents/patent_search/human_eval/doc2vec/patfeats_d2v.npy').item()
target_ids = np.load('/Users/leahelmers/Documents/patent_search/human_eval/doc2vec/target_ids.npy')
random_ids = np.load('/Users/leahelmers/Documents/patent_search/human_eval/doc2vec/random_ids.npy')
dupl_ids = np.load('/Users/leahelmers/Documents/patent_search/human_eval/doc2vec/dupl_ids.npy').item()
cited_ids = np.load('/Users/leahelmers/Documents/patent_search/human_eval/doc2vec/cited_ids.npy').item()

id_dict = make_combis(target_ids, random_ids, cited_ids, dupl_ids)
scores = calc_simcoef_distr(patfeats, ['cited', 'random', 'duplicate'], 
                                      id_dict, 'linear')
auc = calc_auc(scores['cited'], scores['random'])[2]
plot_score_distr('human_eval', 'linear', ['cited', 'duplicate', 'random'], 
                {'cited': scores['cited'], 'duplicate': scores['duplicate'], 'random': scores['random']},
                         auc, ['cited'], histdir='doc2vec_full', bins=50)