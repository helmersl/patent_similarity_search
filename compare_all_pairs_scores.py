import numpy as np
from nlputils.preprocessing import FeatureTransform
from nlputils.simcoefs import compute_sim
from scipy.stats import pearsonr
"""
Calculate the coefficients for all patent pairs the patent attorney
evaluated in all parameter settings and compute correlation (pearson's r)
with human scoring and cited/not cited labels
"""
def calc_correlation(sim_scores, binary_label_pairs, human_label_pairs):
    combis = sim_scores.keys()
    human_scores = [human_label_pairs[combi] for combi in combis]
    cited_scores = [binary_label_pairs[combi] for combi in combis]
    pr_human = pearsonr(human_scores, [sim_scores[combi] for combi in combis])[0]
    pr_cited = pearsonr(cited_scores, [sim_scores[combi] for combi in combis])[0]
    return pr_cited, pr_human

if __name__ == "__main__":
    correlation_simscores_human_cited = {}

    correlation_simscores_human_cited['cited'] = {}
    correlation_simscores_human_cited['human'] = {}
    # load from disk
    combis = np.load('human_eval/corpus_info/combis.npy')
    single_pat_corpus = np.load('human_eval/corpus_info/single_pat_corpus.npy').item()
    binary_label_pairs = np.load('human_eval/corpus_info/binary_label_pairs.npy').item()
    human_label_pairs = np.load('human_eval/corpus_info/human_label_pairs.npy').item()
    for weight in [True, False]:
        weighting = 'None'
        if weight:
            weighting = 'tfidf'
        for norm in ['binary', None]:
            for renorm in ['max', 'length']:
                # make features
                ft = FeatureTransform(identify_bigrams=False, norm=norm, weight=weight, renorm=renorm)
                pat_feats = ft.texts2features(single_pat_corpus)
                # compute scores and calculate AUC for all pairs in combis
                for simcoef in ['linear', 'polynomial', 'sigmoidal', 'histint', 'gaussian',
                    'simpson', 'braun', 'kulczynski', 'jaccard', 'dice', 'otsuka', 'sokal',
                    'manhattan', 'sqeucl', 'minkowski', 'canberra', 'chisq', 'chebyshev', 'hellinger', 'jenshan']:
                    sim_scores = {}
                    for combi in combis:
                        target, pid = combi
                        sim_scores[(target, pid)] = compute_sim(pat_feats[target], pat_feats[pid], simcoef)
                    pr_cited, pr_human = calc_correlation(sim_scores, binary_label_pairs, human_label_pairs)
                    if not np.isnan(pr_cited):
                        correlation_simscores_human_cited['cited']['%s_%s_%s_%s' %(simcoef, str(norm), str(renorm), weighting)] = pr_cited
                    if not np.isnan(pr_human):
                        correlation_simscores_human_cited['human']['%s_%s_%s_%s' %(simcoef, str(norm), str(renorm), weighting)] = pr_human
    np.save('human_eval/correlation_simscores_human_cited.npy', correlation_simscores_human_cited)