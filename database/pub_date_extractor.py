# coding=utf-8
'''
Some patents did not have a correct publication date and thus could not
be inserted into the data base (datetime error)
Their IDs were printed in the nohup.out file. For those patents,
this script extracts the correct publication date by accessing the
webpage again
'''

import re
import random
import urllib2
import pandas as pd
from bs4 import BeautifulSoup as bsoup

def extract_date(patent_id='US20110221676'):
    '''
    This function extracts the correct publication date for a given
    patent ID by accesing its web page again and searching in
    the right place
    '''
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
    publication_date = soup.find_all('td',{'class' : 'single-patent-bibdata'})[4].get_text(separator=u' ')
    return publication_date

def monthrepl(matchobj):
    if matchobj.group(0) == u'Jan':
        return '1'
    elif matchobj.group(0) == u'Febr':
        return '2'
    elif matchobj.group(0) == u'März':
        return '3'
    elif matchobj.group(0) == u'Apr':
        return '4'
    elif matchobj.group(0) == u'Mai':
        return '5'
    elif matchobj.group(0) == u'Juni':
        return '6'
    elif matchobj.group(0) == u'Juli':
        return '7'
    elif matchobj.group(0) == u'Aug':
        return '8'
    elif matchobj.group(0) == u'Sept':
        return '9'
    elif matchobj.group(0) == u'Okt':
        return '10'
    elif matchobj.group(0) == u'Nov':
        return '11'
    elif matchobj.group(0) == u'Dez':
        return '12'

with open('nohup_new.txt') as file_:
    path = '/home/lea/Documents/master_thesis/patent_search/pubdates/'
    pattern= 'patent_*'
    for line in file_:
        pat_id = re.findall('US.*', line)[0]
        pat_id = re.sub('.csv', '', pat_id)
        pub_date = get_pub_date.extract_date(pat_id)
        try:
            assert(re.match('[0-9]{1,2}\. [a-zA-Z\xe4\xc3\xa4]{3,5}\.? [0-9]{4}', pub_date))
        except AssertionError, e:
            print pub_date
            print 'Could not parse: ' + pat_id
        pub_date_datetime = pd.to_datetime(re.sub(
                                                '[a-zA-Z\xe4\xc3\xa4]{3,5}', monthrepl, 
                                                re.sub(' ', '/', re.sub('[.]', '', pub_date))
                                           ),
                                           dayfirst=True)

        df = pd.DataFrame({'id': [pat_id], 'pub_date': [pub_date_datetime]})
        print df
        df.to_csv(path+pat_id+'.csv', sep='\t', encoding='utf-8')
        
