"""
This is a simple helper script for extracting some statistics
on the patents contained in the DB and plotting them
(e.g. average section length, etc.)
"""
import sqlalchemy
import random
import numpy as np
import collections
import matplotlib.pyplot as plt
from database.make_patent_db import load_session, Patent, Citation
from sqlalchemy.sql.expression import func
from nltk.tokenize import word_tokenize


session = load_session()

patent_sample = session.query(Patent).order_by(func.random()).limit(10000)
def calc_section_length():
    section_length = collections.defaultdict(list)
    for pat in patent_sample:
    	section_length['abstract'].append(len(word_tokenize(pat.abstract)))
    	section_length['claims'].append(len(word_tokenize(pat.claims)))
    	section_length['description'].append(len(word_tokenize(pat.description)))
    section_length_avg = {}
    for key in section_length:
        section_length_avg['abstract'] = np.mean(np.array(section_length['abstract']))
        section_length_avg['claims'] = np.mean(np.array(section_length['claims']))
        section_length_avg['description'] = np.mean(np.array(section_length['description']))
    np.save('db_statistics/section_length_avg.npy', section_length_avg)
    np.save('db_statistics/section_length.npy', section_length)

def plot_section_length():
    section_length = np.load('db_statistics/section_length.npy').item()
    description = np.array(section_length['description'])
    claims = np.array(section_length['claims'])
    abstract = np.array(section_length['abstract'])

    plt.hist(claims, bins=50, color='r', histtype='step', label='claims')
    plt.hist(description, bins=50, color='b', histtype='step', label='descriptions')
    plt.hist(abstract, bins=50, color='g', histtype='step', label='abstracts')
    plt.axvline(np.median(claims), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(np.median(abstract), color='g', linestyle='dashed', linewidth=2)
    plt.axvline(np.median(description), color='b', linestyle='dashed', linewidth=2)

    plt.gca().set_xscale("log")
    plt.legend(loc=1)
    plt.xlabel('Number of tokens')
    plt.savefig('db_statistics/section_length.pdf')
    plt.clf()

    claims = np.nan_to_num(np.log(claims))
    abstract = np.nan_to_num(np.log(abstract))
    description = np.nan_to_num(np.log(description))

    from sklearn.neighbors.kde import KernelDensity
    bw = 0.001
    minval=0.
    maxval = np.max(description)
    X_plot = np.linspace(minval, maxval, 50)[:, np.newaxis]
    kde_claim = KernelDensity(kernel='gaussian', bandwidth=bw).fit(claims[:,np.newaxis])
    kde_abstr = KernelDensity(kernel='gaussian', bandwidth=bw).fit(abstract[:,np.newaxis])
    kde_description = KernelDensity(kernel='gaussian', bandwidth=bw).fit(description[:,np.newaxis])
    
    log_dens_claim = kde_claim.score_samples(X_plot)
    log_dens_abstr = kde_abstr.score_samples(X_plot)
    log_dens_description = kde_description.score_samples(X_plot)
    plt.plot(X_plot[:,0], np.exp(log_dens_claim), color='g', label='claims')
    plt.plot(X_plot[:,0], np.exp(log_dens_abstr), color='y', label='abstracts')
    plt.plot(X_plot[:,0], np.exp(log_dens_description), color='b', label='description')

    plt.legend()
    plt.xlabel('Number of tokens')
    plt.savefig('db_statistics/section_length_est.pdf')
    plt.clf()


if __name__ == "__main__":
    calc_section_length()
    plot_section_length()
