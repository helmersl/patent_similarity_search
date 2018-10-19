import pickle as pk
import numpy as np
import pandas as pd
import subprocess
from corpus_utils import PatentCorpus
from plot_utils import plot_score_distr, calc_simcoef_distr, group_combis, calc_auc

def make_input_file(dir_name='/home/lea/Documents/master_thesis/patent_search/database/human_eval/patent_sheets/*'):
    """
    Iterate over patents in single_pat_corpus and write into a dataframe that is then stored in a csv
    """
    # load patent corpus
    pat_corpus = PatentCorpus()
    pat_corpus.mode = 'wmd'
    # ugly hack to invoke __iter__() function :-( :
    list(pat_corpus)
    single_pat_corpus = pat_corpus.single_pat_corpus
    # make empty data frame
    df = pd.DataFrame(columns=['id', 'text'])
    # go through single_pat_corpus and fill data frame
    for pid, text in single_pat_corpus.items():
        df = df.append({'id': pid, 'text': text}, ignore_index=True)
    df.to_csv('human_eval/wmd_pat_corpus.txt', sep='\t', header=False, index=False, encoding='utf-8')

if __name__ == "__main__":
    #make_input_file()
    dist_mat = pk.load(open('/home/lea/Documents/master_thesis/patent_search/wmd-master/dist_matrix.pk'))
    vectors = pk.load(open('/home/lea/Documents/master_thesis/patent_search/wmd-master/vectors.pk'))
    id_list = list(vectors[3])
    combis = np.load('human_eval/corpus_info/combis.npy')
    binary_label_pairs = np.load('human_eval/corpus_info/binary_label_pairs.npy').item()
    human_label_pairs = np.load('human_eval/corpus_info/human_label_pairs.npy').item()
    binary_sim_combis, binary_diff_combis = group_combis(binary_label_pairs)
    human_sim_combis, human_diff_combis = group_combis(human_label_pairs)
    binary_scores = {}
    human_scores = {}
    binary_scores['cited'] = []
    binary_scores['random'] = []
    human_scores['relevant'] = []
    human_scores['not relevant'] = []
    for combi in binary_sim_combis:
        i = id_list.index(combi[0])
        j = id_list.index(combi[1])
        binary_scores['cited'].append(dist_mat[i,j])
    for combi in binary_diff_combis:
        i = id_list.index(combi[0])
        j = id_list.index(combi[1])
        binary_scores['random'].append(dist_mat[i,j])
    for combi in human_sim_combis:
        i = id_list.index(combi[0])
        j = id_list.index(combi[1])
        human_scores['relevant'].append(dist_mat[i,j])
    for combi in human_diff_combis:
        i = id_list.index(combi[0])
        j = id_list.index(combi[1])
        human_scores['not relevant'].append(dist_mat[i,j])
    binary_auc = calc_auc(binary_scores['cited'], binary_scores['random'])[2]
    human_auc = calc_auc(human_scores['relevant'], human_scores['not relevant'])[2]
    plot_score_distr('human_eval', 'linear', ['cited', 'random'], 
                     {'cited': binary_scores['cited'], 'random': binary_scores['random']},
                     binary_auc, ['cited'], histdir='wmd', bins=50)
    plot_score_distr('human_eval', 'linear', ['relevant', 'not relevant'], 
             {'relevant': human_scores['relevant'], 'not relevant': human_scores['not relevant']},
             human_auc, ['relevant'], histdir='wmd', bins=50)