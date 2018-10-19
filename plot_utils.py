import numpy as np
import itertools
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from nlputils.simcoefs import compute_sim
from sklearn.metrics import auc, roc_curve, average_precision_score
#matplotlib.rc('xtick', labelsize=28) 
#matplotlib.rc('ytick', labelsize=28) 
colors = ['#e23a15', '#1a61a7', '#1aa740']

def calc_auc(sim_vals, diff_vals):
    """
    Calculates the AUC value for the separation of target and cited patents

    Input:
        - sim_vals: list of coefficients of the target-cited pairs
        - diff_vals: list of coefficients of the target-random pairs

    Returns:
        - fpr: false positive rates
        - tpr: true positive rates
        - auc_val: area under the curve for separation
    """
    labels = np.hstack((np.array(sim_vals), np.array(diff_vals)))
    true_labels = np.hstack((np.ones(len(sim_vals)), np.zeros(len(diff_vals))))
    fpr, tpr, thresholds = roc_curve(true_labels, labels, pos_label=1)
    auc_val = auc(fpr, tpr)
    aps = average_precision_score(true_labels, labels)
    return fpr, tpr, auc_val, aps

def plot_roc(dir_, simcoef, weighting, norm, renorm, auc_val, fpr, tpr):
    #plot ROC
    print 'plotting ROC'
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' %auc_val)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Score: %s, norm: %s, renorm: %s, AUC: %.2f' %(simcoef, str(norm), str(renorm), auc_val))
    plt.savefig(dir_ + '/rocs/roc_%s_%s_%s_%s.pdf' %(simcoef, str(norm), str(renorm), weighting))
    plt.clf()

def plot_corr(dir_, label_pairs, values):
    cmap=plt.get_cmap('gist_rainbow')
    n_scores = len(label_pairs)
    for i, (score1, score2) in enumerate(label_pairs):
        plt.scatter(values[score1], values[score2])
        axes[i%n_scores,i/n_scores].locator_params(axis = 'x', tight=True, nbins=3)
        axes[i%n_scores,i/n_scores].locator_params(axis = 'y', tight=True, nbins=3)
        a1 = axes[i%n_scores,i/n_scores].scatter(cited1, cited2, rasterized=True, label='cited', s=30)

def plot_score_distr(dir_, simcoef, labels, values, auc_val, args, histdir='', bins=20):
    """
    Plots the sim- and diff-scores distributions

    Inputs:
     - dir_: path to the directory where histograms should be saved (string)
     - simcoef: simcoef the values are calculated with (string)
     - labels: list of strings with the scores to be plotted (e.g.: ['cited', 'random'])
     - values: dictionary containing the values given in lables as keys and the respective
               scores as values
     - auc_val: AUC for the sim- and diff-scores
     - args: list of strings that should be added to title and filename
     - hist_dir: optional extension to hists-directory-name (e.g.: 'bow_directly' or 'd2v')
                 default is empty string
     - bins: Number of bins of the histogram, defaults to 20
    """
    plt.figure(figsize=(16,9))
    for i, label in enumerate(labels):
        plt.hist(values[label], bins=bins, normed=True, histtype='step', label=label, linewidth=3., color=colors[i])
    plt.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3, ncol=len(labels), mode="expand", borderaxespad=0., fontsize=23)
    plt.xlabel('Score', fontsize=26)
    args_str = '_'.join(args)
    #plt.title('Score: %s, %s, AUC: %.2f' % (simcoef, args_str, auc_val), y=1.08)
    #plt.title('AUC: %.4f' % (auc_val), y=1.08)
    plt.title('Score distribution for the cosine similarity (AUC: %.4f)' % (auc_val), y=1.08, fontsize=24)
    #plt.xlim(0,1)
    plt.savefig(dir_ + '/hists_%s/hist_%s_%s.pdf' %(histdir, simcoef, args_str))
    plt.clf()

def group_combis(labels):
    sim_ids = []
    diff_ids = []
    for combi, val in labels.items():
        if val >= 0.5:
            sim_ids.append(combi)
        else:
            diff_ids.append(combi)
    return sim_ids, diff_ids

def make_combis(target_ids, random_ids, cited_ids, dupl_ids):
    id_dict = {}
    id_dict['cited'] = [(key, vals[i]) for key, vals in cited_ids.items() for i in range(len(vals))]
    id_dict['duplicate'] = [(key, vals[i]) for key, vals in dupl_ids.items() for i in range(len(vals))]
    id_dict['random'] = list(itertools.product(cited_ids, random_ids))
    return id_dict

def calc_simcoef_distr(patfeats, labels, id_dict, simcoef):
    """
    Calculates the score distributions

    Inputs:
     - simcoef: simcoef the values are calculated with (string)
     - labels: list of strings with the scores to be calculated (e.g.: ['cited', 'random'])
     - id_dict: dictionary containing the patent ID pairs for the respective label
    Output:
     - scores: dictionary containing the scores for each label
    """
    scores = dict.fromkeys(labels)
    for label in labels:
        print label
        scores[label] = []
        combis = id_dict[label]
        for combi in combis:
            score = compute_sim(patfeats[combi[0]], patfeats[combi[1]], simcoef)
            scores[label].append(score)
    return scores