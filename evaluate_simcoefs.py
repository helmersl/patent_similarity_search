import csv
import re
import datetime
import collections
import sqlalchemy
import random
import numpy as np
import plot_utils
from database.make_patent_db import load_session, Patent, Citation
from nlputils.simcoefs import compute_sim
from nlputils.preprocessing import FeatureTransform
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.schema import Table, MetaData
from sqlalchemy.sql import exists
from sqlalchemy.sql.expression import func

def make_corpus(target_ids, target_pats, random_pats, cited_pats, dupl_pats):
    """
    Make a corpus containing all the patents' raw text

    Input:
        - target_ids: list of target patent IDs
        - target_pats: sql query result for target patents
        - random_pats: sql query result for random patents
        - cited_pats: sql query result for cited patents
        - dupl_pats: sql query result for duplicate patents

    Returns:
        - corpus: dictionary containing patent id as keys and raw text as value
        - random_ids: list of random patent ids
    """
    corpus = {}
    print 'going through random pats'
    for pat in random_pats:
        corpus[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8') + '\n'
                          + pat.description.encode('utf-8') + '\n' + pat.claims.encode('utf-8'))
    random_ids = corpus.keys()
    print 'going through cited pats'
    for pat in cited_pats:
        corpus[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8') + '\n'
                          + pat.description.encode('utf-8') + '\n' + pat.claims.encode('utf-8'))
    print 'going through duplicates'
    for pat in dupl_pats:
        corpus[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8') + '\n'
                          + pat.description.encode('utf-8') + '\n' + pat.claims.encode('utf-8'))
    print 'going through target pats'
    for pat in target_pats:
        corpus[pat.id] = (pat.title.encode('utf-8') + '\n' + pat.abstract.encode('utf-8') + '\n'
                          + pat.description.encode('utf-8') + '\n' + pat.claims.encode('utf-8'))
    return corpus, random_ids

def sample_data(nrand=1000, date=datetime.datetime(2015,1,1,0,0), id_=None, cat=None):
    """
    Extract the target, cited and random patents from the DB

    Input:
        - nrand: Number of random patents to sample (default: 1000)
        - date: Threshold date to separate target and rest patents (default: 01/01/2015)
        - id_: To be set if the scores should be evaluated only for one target patent
               e.g. for the ones scored by patent attorney (default None)
        - cat: To be set if the scores should be evaluated only for a certain category
               e.g. 'A61M'(default None)

    Returns:
        - random_pats:
        - cited_pats:
        - target_pats:
        - dupl_pats:
        - cited_ids:
        - dupl_ids:
    """
    session = load_session()
    # all patents published after given date are considered as target patents
    target_pats = session.query(Patent).filter(Patent.pub_date >= date)
    rest_set = session.query(Patent).filter(Patent.pub_date < date)
    # if the scores are to be calculted for only one target patent
    if id_:
        print "evaluating simscores for patent %s" %id_
        target_pats = session.query(Patent).filter(Patent.id == id_)
    # if the scores should be evaluated only for a certain category 
    elif cat:
        print "evaluating simscores for category %s" %cat
        cat_pats = session.query(Patent).filter(Patent.category.contains(cat))
        cat_pats_ids = [pat.id for pat in cat_pats]
        target_pats = cat_pats.filter(Patent.pub_date >= datetime.datetime(2015,1,1,0,0))
        # the random patents are sampled from the patents published before 2015
        rest_set = cat_pats.filter(Patent.pub_date < datetime.datetime(2015,1,1,0,0))
    else:
        print "evaluating for all target patents"
    # the random patents are sampled from the patents published before given date
    engine = session.get_bind()
    metadata = MetaData()
    metadata.bind = engine
    # create tables for cited and duplicate patents
    cited = Table("cited", metadata,
                  Column('id', String, ForeignKey(Patent.id), primary_key=True))
    duplicates = Table("duplicates", metadata,
                  Column('id', String, ForeignKey(Patent.id), primary_key=True))
    try:
        cited.drop()
        duplicates.drop()
    except:
        pass
    cited.create()
    duplicates.create()
    conn = engine.connect()
    # collect IDs for each target patent
    cited_ids = {}
    dupl_ids = {}
    ## get the duplicates and cited patents for all target patents
    print "getting duplicates and cited patents"
    for patent in target_pats:
        # get duplicate ids(read CSVs)
        with open('/home/lea/Documents/master_thesis/patent_search/pats_2015_apa_lists/apa_list_%s.csv'%str(patent.id)) as apa_file:
            apa_list_reader = csv.reader(apa_file, delimiter='\t')
            duplicates_list = apa_list_reader.next()
            dupl_ids[patent.id] = [unicode(re.sub(' ', '', pat)) for pat in duplicates_list]
        # get cited ids
        citations = session.query(Citation).filter(Citation.citing_pat == patent.id)
        cited_patents = []
        # check, if cited id is a duplicate
        for pat in citations:
            # if the simcoefs are to be evaluated only for a certain category
            if cat:
                # check if the cited pat is in the given category
                if pat.cited_pat  not in cat_pats_ids:
                    continue
            if pat.cited_pat not in dupl_ids[patent.id]:
                cited_patents.append(pat.cited_pat)
        cited_ids[patent.id] = cited_patents
    ## fill tables with cited and duplicate patents
    print "filling tables cited and duplicates"
    # unite all cited and duplicate ids in one list
    all_cited_ids = []
    all_dupl_ids = []
    for pid in cited_ids.keys():
        # fill table with citations
        for cited_id in cited_ids[pid]:
            # check if id equals empty string, if so remove
            if cited_id == u'':
                cited_ids[pid].remove(cited_id)
            # insert patent into table
            else:
                try:
                    ins = cited.insert().values(id=cited_id)
                    conn.execute(ins)
                    all_cited_ids.append(cited_id)
                # error is thrown if patent is already in the DB
                except sqlalchemy.exc.IntegrityError, e:
                    continue
        # fill table with duplicates
        # get duplicate patents for the current target patent
        duplicate_pats = dupl_ids[pid]
        dupls_temp = []
        for dupl_id in duplicate_pats:
            # if the simcoefs are to be evaluated only for a certain category
            if cat:
                # check if the duplicate is in the category
                if pat.cited_pat  not in cat_pats_ids:
                    continue
            # check if id equals empty string
            if dupl_id == u'':
                continue
            # check if the duplicate is already in the DB
            elif session.query(Patent).filter(Patent.id==dupl_id).count() == 0:
                continue
            # insert duplicate patent into duplicates table
            else:
                try:
                    ins = duplicates.insert().values(id=dupl_id)
                    conn.execute(ins)
                    all_dupl_ids.append(dupl_id)
                    dupls_temp.append(dupl_id)
                # error is thrown if patent is already in the DB
                except sqlalchemy.exc.IntegrityError, e:
                    continue
        dupl_ids[pid] = dupls_temp
        assert(len(set(cited_ids[pid]).intersection(set(dupl_ids[pid]))) == 0)
    all_cited_ids = list(set(all_cited_ids))
    all_dupl_ids = list(set(all_dupl_ids))
    # get the cited patents and duplicates
    cited_pats = session.query(Patent).join(cited)
    dupl_pats = session.query(Patent).join(duplicates)
    # sample 1000 random patents not cited by any of the test patents
    print "sampling random patents"
    func.setseed(0.1789)
    candidates = rest_set.order_by(func.random()).limit(len(all_cited_ids) + len(all_dupl_ids) + nrand)
    candidates = candidates.from_self()
    random_pats = candidates.except_(cited_pats)
    random_pats = random_pats.from_self()
    random_pats = random_pats.except_(dupl_pats).order_by(func.random()).limit(nrand)
    random_pats = random_pats.from_self()
    return random_pats, cited_pats, target_pats, dupl_pats, cited_ids, dupl_ids

def calc_simcoef_distr_dict(pat_feats, target_ids, cited_ids, random_ids, dupl_ids, simcoef):
    """
    CAN NOT BE ADAPTED TO EVALUATIN FUNCTION FOR EVALUATE HUMAN SCORES, ETC. BECAUSE WE NEED
    THE STORED DICTS. I THINK IT'S NOT WORTH IT!
    Calculte distribution of simcoef for cited, random and duplicate patents for each target patent

    Input:
        - pat_feats: dictionary containing BOW-dictionaries
        - target_ids: list of target patent IDs
        - cited_ids: dictionary containing list of cited patent IDs for each target patent
        - random_ids: list of random patent ids
        - dupl_ids: dictionary containing list of duplicate patent IDs for each target patent
        - simcoef: coefficient to calculate (string)

    Returns:
        - sim_scores: dictionary containing pairs of patents (target ID, cited ID) as keys
                      and the calculated simcoef as value
        - diff_scores: dictionary containing pairs of patents (target ID, random ID) as keys
                       and the calculated simcoef as value
        - dupl_scores: dictionary containing pairs of patents (target ID, duplicate ID) as keys
                       and the calculated simcoef as value
    """
    sim_scores = {}
    diff_scores = {}
    dupl_scores = {}
    # go through all target patents and compute coefficients
    for pid in target_ids:
        target_pat = pat_feats.get(pid)
        for cid in cited_ids[pid]:
            sim_scores[(pid, cid)] = compute_sim(target_pat, pat_feats[cid], simcoef)
        for sid in random_ids:
            diff_scores[(pid, sid)] = compute_sim(target_pat, pat_feats[sid], simcoef)
        for did in dupl_ids[pid]:
            dupl_scores[(pid, did)] = compute_sim(target_pat, pat_feats[did], simcoef)
    return sim_scores, diff_scores, dupl_scores

def evaluate_coefs(corpus, target_ids, cited_ids, random_ids, dupl_ids, dir_,
                   weights=[True], norms=[None], renorms=['length'], simcoefs=['linear', 'jaccard']): 
    auc_dict = collections.defaultdict(dict) 
    for weight in weights:
        weighting = 'None'
        if weight:
            weighting = 'tfidf'
        auc_dict[weighting] = {}
        for norm in norms:
            auc_dict[weighting][str(norm)] = {}
            for renorm in renorms:
                auc_dict[weighting][str(norm)][str(renorm)] = {}
                # make features
                ft = FeatureTransform(identify_bigrams=False, norm=norm, weight=weight, renorm=renorm)
                pat_feats = ft.texts2features(corpus)
                # compute scores and calculate AUC
                for simcoef in simcoefs:
                    sim_scores, diff_scores, dupl_scores = calc_simcoef_distr_dict(pat_feats, target_ids, cited_ids, random_ids, dupl_ids, simcoef)
                    fpr, tpr, auc_score = plot_utils.calc_auc(sim_scores.values(), diff_scores.values())
                    auc_dict[weighting][str(norm)][str(renorm)][simcoef] = auc_score
                    np.save(dir_ + '/sim_scores/sim_scores_%s_%s_%s_%s.npy' %(simcoef, str(norm), str(renorm), weighting), sim_scores)
                    np.save(dir_ + '/diff_scores/diff_scores_%s_%s_%s_%s.npy' %(simcoef, str(norm), str(renorm), weighting), diff_scores)
                    np.save(dir_ + '/dupl_scores/dupl_scores_%s_%s_%s_%s.npy' %(simcoef, str(norm), str(renorm), weighting), dupl_scores)
                    np.save(dir_ + '/fpr_tpr_rates/fpr_%s_%s_%s_%s.npy' %(simcoef, str(norm), str(renorm), weighting), [fpr, tpr])
                    np.save(dir_ + '/auc_dict.npy', auc_dict)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('from_db', 
                        help='yes, if the corpus and ids should be created using the DB, no, if not',
                        choices=['yes', 'no'])
    parser.add_argument('dir',
                         help='directory path to store sampled data and calculated scores')
    args = parser.parse_args()
    dir_ = args.dir
    if args.from_db == 'yes':
        # sample data
        random_pats, cited_pats, target_pats, dupl_pats, cited_ids, dupl_ids = sample_data(nrand=100,
                                                                                           date=datetime.datetime(2015,1,1,0,0),
                                                                                           id_=None,
                                                                                           cat=None)
        # get all documents
        target_ids = cited_ids.keys()
        corpus, random_ids = make_corpus(target_ids, target_pats, random_pats, cited_pats, dupl_pats)
        assert(sorted(target_ids) == sorted(cited_ids.keys()))
        # save results
        np.save(dir_ + '/corpus.npy', corpus)
        np.save(dir_ + '/target_ids.npy', target_ids)
        np.save(dir_ + '/cited_ids.npy', cited_ids)
        np.save(dir_ + '/random_ids.npy', random_ids)
        np.save(dir_ + '/dupl_ids.npy', dupl_ids)
    if args.from_db == 'no':
        # load previously made corpus and ids
        corpus = np.load(dir_ + '/corpus.npy').item()
        target_ids = np.load(dir_ + '/target_ids.npy')
        cited_ids = np.load(dir_ + '/cited_ids.npy').item()
        random_ids = np.load(dir_ + '/random_ids.npy')
        dupl_ids = np.load(dir_ + '/dupl_ids.npy').item()
    # calculate distributions
    evaluate_coefs(corpus, target_ids, cited_ids, random_ids, dupl_ids, dir_)