import collections
import datetime
import operator
import matplotlib.pyplot as plt
from database.make_patent_db import load_session, Patent, Citation
from sqlalchemy.sql.expression import func

def evaluate_cat_distr(thresh=None, nchars=4):
	"""
	Evaluates the distribution of CPC categories in the data set
	seperately for target patents (published in 2015)
	and the overall data set (patents published between 2000 and 2015)
	and plots the results in a histogram.
	Either a depth reduction or a threshold should be given!

	Inputs:
		- thresh: minimum number of patents a category should contain
				  in order to be considered (default: None)
		- nchars: number of characters that should be considered,
				  defines how deep to step into classification
				  (default: 4 --> e.g.: 'A61M', if set to None, analysis is
				  	very detailed --> e.g.: 'A61M2016/0039')

	"""
	session = load_session()
	# extract target patent set
	target_pats = session.query(Patent).filter(Patent.pub_date >= datetime.datetime(2015,1,1,0,0))
	# draw 10000 random patents from entire population including the target patents
	random_pats = session.query(Patent).order_by(func.random()).limit(10000)

	# get categories for the patents
	target_cats = [pat.category[:nchars] for pat in target_pats]
	target_cat_dict = {k: v for k, v in dict(collections.Counter(target_cats)).items() if v >= thresh}

	random_cats = [pat.category[:nchars] for pat in random_pats]
	random_cat_dict = {k: v for k, v in dict(collections.Counter(random_cats)).items() if v >= thresh}

	# plot category distribution
	fig = plt.figure()
	plt.bar(range(len(target_cat_dict.keys())), target_cat_dict.values())
	plt.xticks(range(len(target_cat_dict)), target_cat_dict.keys(), rotation=70, fontsize=5)
	plt.savefig('db_statistics/target_cat_distr_%s_%s.pdf' %(str(nchars), str(thresh)))
	plt.clf()

	fig = plt.figure()
	plt.bar(range(len(random_cat_dict.keys())), random_cat_dict.values())
	plt.xticks(range(len(random_cat_dict)), random_cat_dict.keys(), rotation=70, fontsize=5)
	plt.savefig('db_statistics/random_cat_distr_%s_%s.pdf' %(str(nchars), str(thresh)))
	plt.clf()

if __name__ == "__main__":
	evaluate_cat_distr(None, 4)
