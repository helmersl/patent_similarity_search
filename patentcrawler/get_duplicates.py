"""
This script extracts the patent ids listed in the
also-published-as-section from google patents
"""

import datetime
import random
import urllib2
import re
import csv
import make_patent_db as mdb
from make_patent_db import Patent
from bs4 import BeautifulSoup as bsoup

session = mdb.load_session()
target_pats = session.query(Patent).filter(Patent.pub_date >= datetime.datetime(2015,1,1,0,0))
def get_apa_list(patent_id='US20110221676'):
    """
    this function extracts the patent ids listed in the
    also-published-as section from google patents for
    the input patent id and saves the list in a csv file
    """
    # make the google patent link
    link = 'https://www.google.de/patents/%s' % patent_id
    # get the html of the webpage containing the original article
    try:
        # random magic browser so you don't immediately get blocked out for making too many requests ;-)
        html = urllib2.urlopen(urllib2.Request(link, headers={'User-Agent': 'Magic Browser%i'%random.randint(0,100)})).read()
    except urllib2.URLError:
        try:
            link = link + 'A1/'
            html = urllib2.urlopen(urllib2.Request(link, headers={'User-Agent': 'Magic Browser%i'%random.randint(0,100)})).read()
        except urllib2.URLError:
            print "Please try again later: URL Error while processing"
            print link
            return None
    # make a beautiful soup object out of the html
    soup = bsoup(html)
    # get also published as - ids
    apa_html = soup.find_all('span',{'class' : 'patent-bibdata-value-list'})
    apa_ids = apa_html[0].get_text(separator=u' ').split(', ')
    # save html
    with open('pats_2015_htmls/%s.html' %patent_id, 'w') as out_html:
        out_html.write(html)
    #save apa_ids in csv
    with open('pats_2015_apa_lists/apa_list_%s.csv' %patent_id, 'w') as out_csv:
        csv_writer = csv.writer(out_csv, delimiter='\t')
        csv_writer.writerow(apa_ids)
    return apa_ids

if __name__ == '__main__':
    for patent in target_pats:
        apa_list = get_apa_list(patent.id)
