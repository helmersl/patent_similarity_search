import random
import urllib2
import re
from bs4 import BeautifulSoup as bsoup
def get_patent(patent_id='US20110221676'):
    # make the google patent link
    link = 'https://www.google.de/patents/%s?cl=de' % patent_id
    print link
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
    # extract patent title
    title = re.split(' - ', soup.find('title').get_text(separator=u' '))[1]
    title = title.encode('utf-8')
    # extract classification (CPC -- for definitions: http://www.cooperativepatentclassification.org/cpcSchemeAndDefinitions/table.html)
    category = None
    categories = soup.find_all('span', {'class':'nested-value'})
    for cat in categories:
        if len(re.findall('A61', cat.get_text())) > 0:
                category = cat.get_text(separator=u' ').encode('utf-8')
                break
    # get patent number to check if publication number is A1
    publication_number = soup.find('span',{'class' : 'patent-number'}).get_text(separator=u' ').encode('utf-8')
    # extract application number (from when it was first submitted) to see if it's native English
    application_number = soup.find_all('td',{'class' : 'single-patent-bibdata'})[2].get_text(separator=u' ').encode('utf-8')
    # extract publication date
    publication_date = soup.find_all('td',{'class' : 'single-patent-bibdata'})[3].get_text(separator=u' ').encode('utf-8')
    # extract the abstract
    abstract = None
    try:
        abstract = soup.find('div', {'class':'abstract'}).get_text(separator=u' ').encode('utf8')
    except:
        abstract = u''
    # extract the description
    description = soup.find('div', {'class':'patent-section patent-description-section'}).get_text(separator=u' ').encode('utf-8')
    # extract the claims
    try:
        claims = soup.find('div', {'class':'claim'}).get_text(separator=u' ').encode('utf-8')
    except:
        return None
    # extract all cited patents - the first table is the one we want
    cited_html = soup.find('table', {'class':'patent-data-table'}).find_all('td', {'class':'patent-data-table-td citation-patent'})
    # extract publication date for the cited patents
    publicationdates_html = soup.find('table', {'class':'patent-data-table'}).find_all('td', {'class':'patent-data-table-td patent-date-value'})[1::2]
    # extract also published as list:
    apa_html = soup.find_all('span',{'class' : 'patent-bibdata-value-list'})
    apa_ids = apa_html[0].get_text(separator=u' ').split(', ')
    cited_patents = []
    for cp in cited_html:
        cited_patents.append(cp.find('a').get_text())
    # Extract year of publication of cited patents
    pub_years = []
    for date in publicationdates_html:
        pub_years.append(date.get_text(separator=u' ')[-4:].encode('utf-8'))
    return title, category, publication_number, application_number, publication_date, abstract, description, claims, cited_patents, pub_years, apa_ids

if __name__ == '__main__':
    patent_id = 'US20110221676'
    print "download information for patent %s" % patent_id
    abstract, cited_patents = get_patent(patent_id)
