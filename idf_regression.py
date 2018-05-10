import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from copy import deepcopy
from database.make_patent_db import load_session, Patent
from corpus_utils import PatentCorpus
from plot_utils import plot_score_distr, calc_simcoef_distr, group_combis, calc_auc
from nlputils.preprocessing import FeatureTransform, features2mat
from nlputils.dict_utils import norm_dict
from nlputils.simcoefs import compute_sim
from sklearn.metrics import auc, roc_curve


def pointwise_dict_multiply(dict1, dict2):
    """
    Inputs:
        - dict1, dict2: dicts with {key: float}
    Returns:
        - dict_mult: dict with {key: dict1[key]*dict2[key]} for all keys dict1 and dict2 have in common
    """
    common_keys = set(dict1.keys()) & set(dict2.keys())
    return {k: dict1[k]*dict2[k] for k in common_keys}

def postprocess_weights(idf_weights, zero, sqrt):
    """
    Inputs:
        - idf_weights: Weights calculated in regression
        - zero: Boolean indicating if negative weights
                should be set to zero
        - sqrt: Boolean indicating if the squareroot should
                be taken for positive weights
    Returns:
        -weights: postprocessed weights
    """
    weights = {}
    for word, weight in idf_weights.items():
        if (weight < 0) & zero:
            weights[word] = 0.
        elif (weight >= 0) & sqrt:
            weights[word] = np.sqrt(weight)
        else:
            weights[word] = weight
    return weights

def model_selection(combis, patfeats_pairs, single_pat_corpus, binary_label_pairs, human_label_pairs):
    alphas = np.arange(10)/100000.
    param_auc_dict = {}
    param_auc_dict['cited'] = {}
    param_auc_dict['human'] = {}
    for alpha in alphas:
        param_auc_dict['cited']['%.5f' %alpha] = {}
        param_auc_dict['human']['%.5f' %alpha] = {}
        for wtype in ['idf_weights', 'idf_weights_sqrt', 'idf_weights_zeroed', 'idf_weights_zeroed_sqrt']:
            param_auc_dict['cited']['%.5f' %alpha][wtype] = []
            param_auc_dict['human']['%.5f' %alpha][wtype] = []
    ## model selection
    for n in range(5):
        print "testing for the %ith time" %n
        # train/test split
        combis_perm = np.random.permutation(combis)
        trainids = combis_perm[:int(np.ceil(len(combis)*0.7))]
        testids = combis_perm[int(np.ceil(len(combis)*0.7)):]
        patfeats_pairs_train = {}
        for combi in trainids:
            target_id, pid = combi
            patfeats_pairs_train[(target_id, pid)] = patfeats_pairs[(target_id, pid)]
        train_pair_ids = patfeats_pairs_train.keys()
        # transform into feature matrix (number of pairs) x (bow-dim)
        print "make feature matrix train"
        featmat_train, featurenames = features2mat(patfeats_pairs_train, train_pair_ids)
        # same for test set
        patfeats_pairs_test = {}
        for combi in testids:
            target_id, pid = combi
            patfeats_pairs_test[(target_id, pid)] = patfeats_pairs[(target_id, pid)]
        test_pair_ids = patfeats_pairs_test.keys()
        print "make feature matrix test"
        featmat_test, featurenames = features2mat(patfeats_pairs_test, test_pair_ids, featurenames)

        # get the corresponding label vectors
        y_human_train = [human_label_pairs[tid] for tid in train_pair_ids]        
        y_human_test = [human_label_pairs[tid] for tid in test_pair_ids]
        y_binary_train = [binary_label_pairs[tid] for tid in train_pair_ids]
        y_binary_test = [binary_label_pairs[tid] for tid in test_pair_ids]

        for alpha in alphas:
            # perform the linear regression for binary (cited/not cited) labels
            print "perform regression for binary scoring"
            clf = lm.Lasso(alpha=alpha, fit_intercept=True, random_state=13)
            clf.fit(featmat_train, y_binary_train)
            ## calculate AUC-values
            # the fitted coefficients are now our word weights
            # perform regression for all weight postprocessings
            weights = {}
            weights['idf_weights'] = norm_dict(dict(zip(featurenames, clf.coef_)))
            weights['idf_weights_zeroed'] = postprocess_weights(weights['idf_weights'], zero=True, sqrt=False)
            weights['idf_weights_sqrt'] = postprocess_weights(weights['idf_weights'], zero=False, sqrt=False)
            weights['idf_weights_zeroed_sqrt'] = postprocess_weights(weights['idf_weights'], zero=True, sqrt=True)

            # multiply patfeats with idf weights
            for wtype in ['idf_weights', 'idf_weights_sqrt', 'idf_weights_zeroed', 'idf_weights_zeroed_sqrt']:
                ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
                ft.Dw = weights[wtype]
                patfeats_idf = ft.texts2features(single_pat_corpus)

                # calculate auc for cited/not cited on test set
                for simcoef in ['linear']:
                    y_true = []
                    y_pred = []
                    for combi in testids:
                        y_true.append(binary_label_pairs[(combi[0], combi[1])]) 
                        y_pred.append(compute_sim(patfeats_idf[combi[0]], patfeats_idf[combi[1]], simcoef))
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                    auc_val = auc(fpr, tpr)
                    print "cited, alpha: %.5f, AUC: %.4f" %(alpha, auc_val)
                    param_auc_dict['cited']['%.5f' %alpha][wtype].append(auc_val)

            print "perform regression for human scoring"
            clf = lm.Lasso(alpha=alpha, fit_intercept=True, random_state=13)
            clf.fit(featmat_train, y_human_train)
            ## calculate AUC-values
            # the fitted coefficients are now our word weights
            # perform regression for all weight postprocessings
            weights = {}
            weights['idf_weights'] = norm_dict(dict(zip(featurenames, clf.coef_)))
            weights['idf_weights_zeroed'] = postprocess_weights(weights['idf_weights'], zero=True, sqrt=False)
            weights['idf_weights_sqrt'] = postprocess_weights(weights['idf_weights'], zero=False, sqrt=False)
            weights['idf_weights_zeroed_sqrt'] = postprocess_weights(weights['idf_weights'], zero=True, sqrt=True)

            # multiply patfeats with idf weights
            for wtype in ['idf_weights', 'idf_weights_sqrt', 'idf_weights_zeroed', 'idf_weights_zeroed_sqrt']:
                ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
                ft.Dw = weights[wtype]
                patfeats_idf = ft.texts2features(single_pat_corpus)
                
                # calculate auc for cited/not cited on test set
                for simcoef in ['linear']:
                    y_true = []
                    y_pred = []
                    for combi in testids:
                        y_true.append(int(human_label_pairs[(combi[0], combi[1])] >= 0.5)) 
                        y_pred.append(compute_sim(patfeats_idf[combi[0]], patfeats_idf[combi[1]], simcoef))
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
                    auc_val = auc(fpr, tpr)
                    print "human, alpha: %.5f, AUC: %.4f" %(alpha, auc_val)
                    param_auc_dict['human']['%.5f' %alpha][wtype].append(auc_val)
    np.save('human_eval/regression/param_auc_dict.npy', param_auc_dict)

if __name__ == "__main__":
    pat_corpus = PatentCorpus()
    pat_corpus.mode = 'regression'
    # ugly hack to invoke __iter__() function :-( :
    list(pat_corpus)
    pat_ids = pat_corpus.pat_ids
    combis = pat_corpus.combis
    binary_label_pairs = pat_corpus.binary_label_pairs
    human_label_pairs = pat_corpus.human_label_pairs
    single_pat_corpus = pat_corpus.single_pat_corpus
    ## baseline: cosine similarity calculation with idf weights
    print "baseline: idf weights"
    # make features
    ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
    patfeats_org = ft.texts2features(single_pat_corpus)
    #save corpus and ids for training word2vec
    np.save('human_eval/corpus_info/pat_ids.npy', pat_ids)
    np.save('human_eval/corpus_info/combis.npy', combis)
    np.save('human_eval/corpus_info/binary_label_pairs.npy', binary_label_pairs)
    np.save('human_eval/corpus_info/human_label_pairs.npy', human_label_pairs)
    np.save('human_eval/corpus_info/patfeats_human_eval.npy', patfeats_org)
    np.save('human_eval/corpus_info/single_pat_corpus.npy', single_pat_corpus)
    # save the idf weights
    Dw_all = {}
    Dw_all['idf'] = deepcopy(ft.Dw)

    ## our case: weights learned by regression  
    # transform into very basic features, i.e. w/o idf weights
    print "making patent pair features"
    ft = FeatureTransform(identify_bigrams=False, norm=None, weight=False, renorm=None)
    # transform into pair features + baseline cosine labels
    patfeats = ft.texts2features(single_pat_corpus)
    # make pairwise feature matrix
    print "making feature matrix"
    patfeats_pairs = {}
    binary_labels = []
    human_labels = []
    for combi in combis:
        target_id, pid = combi
        binary_labels.append(binary_label_pairs[(target_id, pid)])
        #if performing logistic regression, uncomment the following line and comment the one after that:
        #human_labels.append(int(human_label_pairs[(target_id, pid)] >=0.5))
        human_labels.append(human_label_pairs[(target_id, pid)])
        patfeats_pairs[(target_id, pid)] = norm_dict(pointwise_dict_multiply(patfeats[target_id], patfeats[pid]), 'length')
    featmat, featurenames = features2mat(patfeats_pairs, combis)
    # perform model selection
    #print "performing model selection"
    #model_selection(combis, patfeats_pairs, single_pat_corpus, binary_label_pairs, human_label_pairs)
    # preform regression for cited/not cited labels
    print "performing regression for cited/not cited"
    # if logistic regression should be performed, uncomment the following line
    #clf = lm.LogisticRegression(C=1.5, fit_intercept=True, solver='liblinear', random_state=13)
    clf = lm.Lasso(alpha=0.00005, fit_intercept=True, random_state=13)
    clf.fit(featmat, binary_labels)
    idf_weights = norm_dict(dict(zip(featurenames, clf.coef_)))
    weights = postprocess_weights(idf_weights, zero=True, sqrt=False)
    Dw_all['binary'] = weights

    # preform regression for relevant/not relevant (human) labels
    print "performing regression for relevant/not relevant"
    # if logistic regression should be performed, uncomment the following line
    #clf = lm.LogisticRegression(C=1.5, fit_intercept=True, solver='liblinear', random_state=13)
    clf = lm.Lasso(alpha=0.00006, fit_intercept=True, random_state=13)
    clf.fit(featmat, human_labels)
    idf_weights = norm_dict(dict(zip(featurenames, clf.coef_)))
    weights = postprocess_weights(idf_weights, zero=True, sqrt=False)
    Dw_all['human'] = weights

    # combine normally calculated idf weights with regression idf weights
    Dw_all['human_idf'] = pointwise_dict_multiply(Dw_all['idf'], Dw_all['human'])
    Dw_all['binary_idf'] = pointwise_dict_multiply(Dw_all['idf'], Dw_all['binary'])
    np.save('human_eval/regression/Dw_all.npy', Dw_all)
    
    ## plot all the word weights against each other to see the differences
    #assert set(Dw_all['idf'].keys()) == set(Dw_all['reg_sanity'].keys())
    wordlist = sorted(set(Dw_all['idf'].keys()) & set(Dw_all['binary'].keys()))
    assert wordlist == sorted(set(Dw_all['human'].keys()))
    idf_weights = [Dw_all['idf'][w] for w in wordlist]
    binary_weights = [Dw_all['binary'][w] for w in wordlist]
    human_weights = [Dw_all['human'][w] for w in wordlist]

    plt.figure()
    plt.scatter(idf_weights, human_weights)
    plt.xlabel('idf weights')
    plt.ylabel('human weights')
    plt.savefig('human_eval/regression/idf_human.pdf')
    plt.close()

    plt.figure()
    plt.scatter(idf_weights, binary_weights)
    plt.xlabel('idf weights')
    plt.ylabel('binary weights')
    plt.savefig('human_eval/regression/idf_binary.pdf')
    plt.close()

    plt.figure()
    plt.scatter(idf_weights, binary_weights)
    plt.xlabel('human weights')
    plt.ylabel('binary weights')
    plt.savefig('human_eval/regression/human_binary.pdf')
    plt.close()
    
    # multiply patfeats with binary idf weights
    ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
    ft.Dw = Dw_all['binary_idf']
    patfeats_cited = ft.texts2features(single_pat_corpus)
    # multiply patfeats with human idf weights
    ft = FeatureTransform(identify_bigrams=False, norm=None, weight=True, renorm='length')
    ft.Dw = Dw_all['human_idf']
    patfeats_human = ft.texts2features(single_pat_corpus)
    # plot the distributions
    binary_sim_combis, binary_diff_combis = group_combis(binary_label_pairs)
    human_sim_combis, human_diff_combis = group_combis(human_label_pairs)
    for simcoef in ['linear', 'jaccard']:
        binary_scores = calc_simcoef_distr(patfeats_cited, ['cited', 'random'], 
                                           {'cited': binary_sim_combis, 'random': binary_diff_combis},
                                           simcoef)
        human_scores = calc_simcoef_distr(patfeats_human, ['relevant', 'not relevant'],
                                          {'relevant': human_sim_combis, 'not relevant': human_diff_combis},
                                          simcoef)
        binary_auc = calc_auc(binary_scores['cited'], binary_scores['random'])[2]
        human_auc = calc_auc(human_scores['relevant'], human_scores['not relevant'])[2]
        plot_score_distr('human_eval', simcoef, ['cited', 'random'], 
                         {'cited': binary_scores['cited'], 'random': binary_scores['random']},
                         binary_auc, ['cited'], histdir='reg_idf', bins=50)
        plot_score_distr('human_eval', simcoef, ['relevant', 'not relevant'], 
                 {'relevant': human_scores['relevant'], 'not relevant': human_scores['not relevant']},
                 human_auc, ['relevant'], histdir='reg_idf', bins=50)