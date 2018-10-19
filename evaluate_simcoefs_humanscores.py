import glob
import collections
import itertools
import numpy as np
import pandas as pd
from plot_utils import plot_score_distr, calc_simcoef_distr, group_combis, calc_auc
from corpus_utils import PatentCorpus
from nlputils.preprocessing import FeatureTransform

def evaluate_from_df():
    ## create data structures for calculating values directly from sheets
    auc_vals = collections.defaultdict(dict)
    sim_scores = collections.defaultdict(dict)
    diff_scores = collections.defaultdict(dict)

    sim_scores['cited']['cosine'] = []
    diff_scores['cited']['cosine'] = []
    sim_scores['cited']['jaccard'] = []
    diff_scores['cited']['jaccard'] = []

    sim_scores['human']['cosine'] = []
    diff_scores['human']['cosine'] = []
    sim_scores['human']['jaccard'] = []
    diff_scores['human']['jaccard'] = []

    # access all 10 evaluation sheets to extract scores and labels
    for csv_sheet in glob.glob('human_eval/patent_sheets/*'):
        # read the csv sheet
        pat_df = pd.read_csv(csv_sheet, delimiter='\t', usecols=['id', 'cited', 'human', 'cosine', 'jaccard'])
        # drop rows with missing values
        pat_df = pat_df.dropna()
        pat_df = pat_df.reset_index()
        pat_df.human = pat_df.human/5.

        ## Extract scores from data frame
        sim_scores['cited']['cosine'].extend(list(np.array(pat_df.cosine)[np.array(pat_df.cited).astype(int).nonzero()]))
        diff_scores['cited']['cosine'].extend(list(np.array(pat_df.cosine)[(np.array(pat_df.cited).astype(int) == 0).nonzero()]))
        sim_scores['cited']['jaccard'].extend(list(np.array(pat_df.jaccard)[np.array(pat_df.cited).astype(int).nonzero()]))
        diff_scores['cited']['jaccard'].extend(list(np.array(pat_df.jaccard)[(np.array(pat_df.cited).astype(int) == 0).nonzero()]))

        sim_scores['human']['cosine'].extend(list(np.array(pat_df.cosine)[np.array(pat_df.human > 0.5).astype(int).nonzero()]))
        diff_scores['human']['cosine'].extend(list(np.array(pat_df.cosine)[(np.array(pat_df.human > 0.5).astype(int) == 0).nonzero()]))
        sim_scores['human']['jaccard'].extend(list(np.array(pat_df.jaccard)[np.array(pat_df.human > 0.5).astype(int).nonzero()]))
        diff_scores['human']['jaccard'].extend(list(np.array(pat_df.jaccard)[(np.array(pat_df.human > 0.5).astype(int) == 0).nonzero()]))

    auc_vals['cited']['cosine'] = calc_auc(sim_scores['cited']['cosine'], diff_scores['cited']['cosine'])[2]
    auc_vals['cited']['jaccard'] = calc_auc(sim_scores['cited']['jaccard'], diff_scores['cited']['jaccard'])[2]
    auc_vals['human']['cosine'] = calc_auc(sim_scores['human']['cosine'], diff_scores['human']['cosine'])[2]
    auc_vals['human']['jaccard'] = calc_auc(sim_scores['human']['cosine'], diff_scores['human']['cosine'])[2]
    # plot results
    for score, coef in itertools.product(['human','cited'], ['jaccard','cosine']):
        if score == 'cited':
            labels = ['random', 'cited']
        elif score == 'human':
            labels = ['irrelevant', 'relevant']
        else:
            print "Score is not known!"
        plot_score_distr('human_eval', coef, labels, {labels[1]: sim_scores[score][coef], labels[0]: diff_scores[score][coef]}, 
                         auc_vals[score][coef], [labels[1]], 'bow_directly')

def evaluate_from_db():
    ## Sanity Check: Make the plots calculating the values again
    # get patent corpus
    pat_corpus = PatentCorpus()
    pat_corpus.mode = 'regression'
    # ugly hack to invoke __iter__() function :-( :
    list(pat_corpus)
    # get relevant information
    pat_ids = pat_corpus.pat_ids
    combis = pat_corpus.combis
    binary_label_pairs = pat_corpus.binary_label_pairs
    human_label_pairs = pat_corpus.human_label_pairs
    single_pat_corpus = pat_corpus.single_pat_corpus
    # make dict of feature vectors
    ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
    patfeats = ft.texts2features(single_pat_corpus)
    # calculate scores and plot them
    # separate combis into similar pairs and different pairs according to the binary and human labels
    binary_sim_combis, binary_diff_combis = group_combis(binary_label_pairs)
    human_sim_combis, human_diff_combis = group_combis(human_label_pairs)
    # calculate sim- and diff-scores and AUC-values and plot scores, saved in human_eval/hists_bow
    for simcoef in ['linear', 'jaccard']:
        binary_scores = calc_simcoef_distr(patfeats, ['random', 'cited'], 
                                           {'cited': binary_sim_combis, 'random': binary_diff_combis},
                                           simcoef)
        human_scores = calc_simcoef_distr(patfeats, ['irrelevant', 'relevant'],
                                          {'relevant': human_sim_combis, 'irrelevant': human_diff_combis},
                                          simcoef)
        binary_auc = calc_auc(binary_scores['cited'], binary_scores['random'])[2]
        human_auc = calc_auc(human_scores['relevant'], human_scores['irrelevant'])[2]
        plot_score_distr('human_eval', simcoef, ['random', 'cited'], 
                         {'cited': binary_scores['cited'], 'random': binary_scores['random']},
                         binary_auc, ['cited'], histdir='bow', bins=50)
        plot_score_distr('human_eval', simcoef, ['irrelevant', 'relevant'], 
                 {'relevant': human_scores['relevant'], 'irrelevant': human_scores['irrelevant']},
                 human_auc, ['relevant'], histdir='bow', bins=50)

if __name__ == "__main__":
    evaluate_from_df()
    evaluate_from_db()