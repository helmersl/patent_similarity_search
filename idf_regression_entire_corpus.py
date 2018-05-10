import itertools
import numpy as np
import sklearn.linear_model as lm
from sklearn.tree import DecisionTreeRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from plot_utils import plot_score_distr, calc_simcoef_distr, group_combis, calc_auc
from nlputils.preprocessing import FeatureTransform, features2mat
from nlputils.dict_utils import norm_dict
from nlputils.ml_utils import xval

def pointwise_dict_multiply(dict1, dict2):
    '''
    Inputs:
        - dict1, dict2: dicts with {key: float}
    Returns:
        - dict_mult: dict with {key: dict1[key]*dict2[key]} for all keys dict1 and dict2 have in common
    '''
    common_keys = set(dict1.keys()) & set(dict2.keys())
    return {k: dict1[k]*dict2[k] for k in common_keys}

def postprocess_weights(idf_weights, zero, sqrt):
    '''
    Inputs:
        - idf_weights: Weights calculated in regression
        - zero: Boolean indicating if negative weights
                should be set to zero
        - sqrt: Boolean indicating if the squareroot should
                be taken for positive weights
    Returns:
        - weights: postprocessed weights
    '''
    weights = {}
    for word, weight in idf_weights.items():
        if (weight < 0) & zero:
            weights[word] = 0.
        elif (weight >= 0) & sqrt:
            weights[word] = np.sqrt(weight)
        else:
            weights[word] = weight
    return weights

if __name__ == "__main__":
    #Dw_all = {}
    Dw_all = np.load('full_patent_scores/corpus_info_for_regression/Dw_all.npy').item()
    #load corpus from disk
    pat_corpus = np.load('full_patent_scores/corpus_info_for_regression/corpus.npy').item()
    cited_ids = np.load('full_patent_scores/corpus_info_for_regression/cited_ids.npy').item()
    random_ids = np.load('full_patent_scores/corpus_info_for_regression/random_ids.npy')
    #take only half of the random patents into account to avoid memory error
    random_ids = random_ids[np.random.choice(range(len(random_ids)), len(random_ids)/2.)]
    sim_combis = []
    diff_combis = []
    # join the ids for sim- and diff-pairs
    for target_pat, cited_pats in cited_ids.items():
        #get diff scores
        diff_combis.extend([target_pat + '_' + rid for rid in random_ids])
        #get sim scores
        sim_combis.extend([target_pat + '_' + cid for cid in cited_pats])
    #save to disk
    np.save('full_patent_scores/corpus_info_for_regression/sim_combis.npy', sim_combis)
    np.save('full_patent_scores/corpus_info_for_regression/diff_combis.npy', diff_combis)


    labels = np.hstack((np.ones(len(sim_combis)), np.zeros(len(diff_combis))))
    combis = np.array(sim_combis + diff_combis)
    '''
    ## baseline: cosine similarity calculation with idf weights 
    print "baseline: idf weights"
    # make features
    ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
    patfeats_org = ft.texts2features(pat_corpus)
    
    # save the idf weights
    Dw_all['idf'] = deepcopy(ft.Dw)
    np.save('full_patent_scores/corpus_info_for_regression/Dw_all.npy', Dw_all)
    '''
    ## our case: weights learned by regression  
    # transform into very basic features, i.e. w/o idf weights
    print "making patent pair features"
    ft = FeatureTransform(identify_bigrams=False, norm=None, weight=False, renorm=None)
    # transform into pair features + baseline cosine labels
    patfeats = ft.texts2features(pat_corpus)
    # make pairwise feature matrix
    print "making feature matrix"
    patfeats_pairs = {}
    for combi in combis:
        target_id, pid = combi.split('_')
        patfeats_pairs[target_id + '_' + pid] = norm_dict(pointwise_dict_multiply(patfeats[target_id], patfeats[pid]), 'length')
    featmat, featurenames = features2mat(patfeats_pairs, combis)
    '''
    print "performing regression"
    # perform logistig regression
    log_reg = lm.LogisticRegression(C=1., fit_intercept=True, solver='liblinear', random_state=13)
    log_reg.fit(featmat, labels)
    weights_logreg = norm_dict(dict(zip(featurenames, log_reg.coef_)))
    Dw_all['logreg'] = weights_logreg
    '''
    # perform regression with lasso
    clf = lm.Lasso(alpha=0.00005, fit_intercept=True, random_state=13)
    clf.fit(featmat, labels)
    idf_weights = norm_dict(dict(zip(featurenames, clf.coef_)))
    weights = postprocess_weights(idf_weights, zero=True, sqrt=False)
    Dw_all['lasso'] = weights
    Dw_all['lasso_neg'] = idf_weights
    np.save('full_patent_scores/corpus_info_for_regression/Dw_all.npy', Dw_all)
    '''
    # perform Ridge regression
    clf = lm.Ridge(alpha=0.00005, fit_intercept=True, random_state=13)
    clf.fit(featmat, labels)
    idf_weights = norm_dict(dict(zip(featurenames, clf.coef_)))
    weights = postprocess_weights(idf_weights, zero=True, sqrt=False)
    Dw_all['ridge'] = weights
    Dw_all['ridge_neg'] = idf_weights
    # perform linear regression
    clf = lm.LinearRegression(fit_intercept=True, normalize=True)
    clf.fit(featmat, labels)
    idf_weights = norm_dict(dict(zip(featurenames, clf.coef_)))
    weights = postprocess_weights(idf_weights, zero=True, sqrt=False)
    Dw_all['linreg'] = weights
    Dw_all['linreg_neg'] = idf_weights
    # perform regression with decision tree
    clf = DecisionTreeRegressor(min_samples_leaf=10, random_state=13)
    clf.fit(featmat, labels)
    idf_weights = norm_dict(dict(zip(featurenames, clf.feature_importances_)))
    #weights = postprocess_weights(idf_weights, zero=True, sqrt=False)
    Dw_all['dec_tree'] = weights
    
    Dw_all['binary_idf'] = pointwise_dict_multiply(Dw_all['idf'], Dw_all['binary'])
    np.save('full_patent_scores/corpus_info_for_regression/Dw_all.npy', Dw_all)
    Dw_all['binary_idf_neg'] = pointwise_dict_multiply(Dw_all['idf'], Dw_all['binary_neg'])
    np.save('full_patent_scores/corpus_info_for_regression/Dw_all.npy', Dw_all)
    '''
    for method in ['lasso']:
        ## use learned term weights for feature extraction
        ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
        ft.Dw = Dw_all['lasso']
        patfeats = ft.texts2features(pat_corpus)
        # plot the results
        for simcoef in ['linear']:
            binary_scores = calc_simcoef_distr(patfeats, ['cited', 'random'], 
                                               {'cited': [sim_combi.split('_') for sim_combi in sim_combis], 
                                               'random': [diff_combi.split('_') for diff_combi in diff_combis]},
                                               simcoef)
            binary_auc = calc_auc(binary_scores['cited'], binary_scores['random'])[2]
            plot_score_distr('full_patent_scores', simcoef, ['cited', 'random'], 
                             {'cited': binary_scores['cited'], 'random': binary_scores['random']},
                             binary_auc, ['cited'], histdir=method, bins=50)




