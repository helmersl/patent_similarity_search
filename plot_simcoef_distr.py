'''
This script plots the histogram and roc curves for the evaluated
distance measures
'''
import numpy as np
from plot_utils import plot_score_distr, plot_roc

def plot_distr(dir_, weights=[True], norms=[None], renorms=['length'], simcoefs=['linear'], dupl=True, roc=False):
    auc_dict = np.load(dir_ + '/auc_dict.npy').item()
    # go through norm combinations...
    for weight in weights:
        weighting = 'None'
        if weight:
            weighting = 'tfidf'
        for norm in norms:
            for renorm in renorms:
                for simcoef in simcoefs:
                    try:
                        sim_scores = np.load(dir_ + '/sim_scores/sim_scores_%s_%s_%s_%s.npy' 
                                                    %(simcoef, str(norm), str(renorm), weighting)).item()
                        diff_scores = np.load(dir_ + '/diff_scores/diff_scores_%s_%s_%s_%s.npy' 
                                                     %(simcoef, str(norm), str(renorm), weighting)).item()
                        # if duplicate sim scores are to be plotted
                        if dupl:
                            dupl_scores = np.load(dir_ + '/dupl_scores/dupl_scores_%s_%s_%s_%s.npy' 
                                                         %(simcoef, str(norm), str(renorm), weighting)).item()
                        auc_val = auc_dict[weighting][str(norm)][str(renorm)][simcoef]
                        # if roc curve should be plotted
                        if roc:
                            [fpr, tpr] = np.load(dir_ + '/fpr_tpr_rates/fpr_%s_%s_%s_%s.npy' 
                                               %(simcoef, str(norm), str(renorm), weighting))
                    except:
                        print "data missing for: %s, %s, %s, %s"  %(simcoef, str(norm), str(renorm), weighting)
                        continue
                    try:
                        if dupl:
                            plot_score_distr(dir_, simcoef, ['random', 'cited', 'duplicate'], {'cited': sim_scores.values(), 
                                             'random': diff_scores.values(), 'duplicate': dupl_scores.values()}, 
                                             auc_val, [str(weighting), str(norm), str(renorm)], bins = 80)
                        else:
                            plot_score_distr(dir_, simcoef, ['random', 'cited'], {'cited': sim_scores.values(), 
                                             'random': diff_scores.values(),}, 
                                             auc_val, [str(weighting), str(norm), str(renorm)], bins = 80)
                        if roc:
                            plot_roc(dir_, simcoef, weighting, norm, renorm, auc_val, fpr, tpr)
                    except:
                        print 'could not plot %s, %s, %s' %(simcoef, str(norm), str(renorm))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                         help='directory path to store sampled data and calculated scores')
    parser.add_argument('--roc',
                         help='if roc curves should be plotted')
    args = parser.parse_args()
    dir_ = args.dir
    plot_distr(dir_, dupl=True, roc=args.roc)