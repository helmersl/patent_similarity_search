import numpy as np
from plot_utils import plot_score_distr, group_combis, calc_auc


binary_label_pairs = np.load('human_eval/corpus_info/binary_label_pairs.npy').item()
human_label_pairs = np.load('human_eval/corpus_info/human_label_pairs.npy').item()
combis = np.load('human_eval/corpus_info/combis.npy')
human_sim_combis, human_diff_combis = group_combis(human_label_pairs)
sim_vals = [binary_label_pairs[combi] for combi in human_sim_combis]
diff_vals = [binary_label_pairs[combi] for combi in human_diff_combis]
fpr, tpr, auc_val = calc_auc(sim_vals, diff_vals)
plot_score_distr('human_eval', 'cited', ['relevant', 'not relevant'], 
                 {'relevant': sim_vals, 'not relevant': diff_vals},
                 auc_val, ['relevant'], histdir='baseline', bins=10)