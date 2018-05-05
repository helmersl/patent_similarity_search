import scrape_patents
import numpy as np
import pandas as pd
import time
import datetime
'''
This scrpit creates a data set consisting of (hopefully native) English patent texts published
at google.patents.com with the following properties:
    - publication year: 2000 and onwards
    - Published in the US, Australia, Great Britain, Ireland, Canada or South Africa (to assure native English)
    - Category A61 (medicine/health)
    The following condition was discarded because it didn't seem to be necessary!
    - Application number A1, i.e. search report is included (relevant patents)
'''
# start with some seed patent ids
#seed_patents = [US2015306149, US2015306103, US2015306092]
# register the ids of already fetched patents
#fetched_patents = []
def run(seed_patents, fetched_patents, visited_patents, patents_that_will_be_fetched):
    for pid in seed_patents:
        # list for the cited patents that will be included in our data set
        cited_pats_to_add = []
        if pid not in fetched_patents:
            try:
                title, category, publication_number, application_number, publication_date, abstract, description, claims, cited_patents, pub_years, apa_ids = scrape_patents.get_patent(pid)
            except TypeError:
                print 'patent with id %s was not found' %pid
                pass
            print pid
            print category
            # check if the patent fulfills all the requirements for being included in our data set
            #if int(publication_date[-4:]) >= 2000 and application_number[:2] in [u'US', u'CA', u'GB', u'AU', u'IE', u'ZA'] and category[:3] == u'A61':
            if int(publication_date[-4:]) >= 2000 and application_number[:2] == u'DE' and category[:3] == u'A61':
                # check, which of the cited patents will be included in the data set and are the desired output of similar patents for pid
                for cpid in cited_patents:
                    if cpid in apa_ids:
                        continue
                    if fetch_cited_patent(cpid):
                        cited_pats_to_add.append(cpid)
                # finally, write all the info into a csv file
                fetched_patents.append(pid)
                path_to_logfile='/home/lhelmers/patent_search/patentcrawler/patentcrawler_de.log'
                f = open(path_to_logfile, 'w')
                f.write('fetched_patents = ' + repr(fetched_patents) + '\n')
                f.close()
                dump_to_csv(pid, [title, category, publication_number, application_number, publication_date, abstract, description, claims, cited_pats_to_add, pub_years, apa_ids])

def fetch_cited_patent(patent_id):
    '''
    This is a recursive function that checks for each passed patent id if it fulfills
    the requirements for being included in the data set. It returns a boolean value
    such that the calling patent knows if this patent will be included in the data set.
    If it fulfills the requirements, it is dumped into a csv and the funcion is called
    again for its cited patents (-> recursion)

    input:
        patent_id : id of the patent that has to be checked
    output:
        to_add : boolean value saying if the patent fulfills the requirements (True) or not (False)
    '''
    print "entering fetch_cited_patents for %s" %patent_id
    print len(fetched_patents), len(visited_patents), len(patents_that_will_be_fetched)
    if len(patents_that_will_be_fetched) > 1000:
        return False
    if len(fetched_patents)%500 == 0 and len(fetched_patents) > 0:
        print str(datetime.date.today()) + ' -- ' + time.strftime("%H:%M:%S")
        print '%i patents collected' % len(fetched_patents)
        print '\n'
    to_add = False
    if patent_id in fetched_patents:
        # if the patent has already been fetched, no need to extract all the info again, just return that
        # the calling patent should include it in its cited_patents list
        to_add = True
    elif patent_id in visited_patents:
        print 'patent already visited, ending recursion for %s' %patent_id
        return to_add
    else:
        visited_patents.append(patent_id)
        # if the patent hasn't been fetched yet, check, if it fulfills requirements
        cited_patents_to_add = []
        try:
            title, category, publication_number, application_number, publication_date, abstract, description, claims, cited_patents, pub_years, apa_ids = scrape_patents.get_patent(patent_id)
            entries = title, category, publication_number, application_number, publication_date, abstract, description, claims, cited_patents, pub_years, apa_ids
            if entries.count(None) > 0:
                to_add = False
                print 'returning for patent id %s, does not fulfill requirements' %patent_id
                return to_add
        except:
            to_add = False
            return to_add
        #to_add = int(publication_date[-4:]) >= 2000 and application_number[:2] in [u'US', u'CA', u'GB', u'AU', u'IE', u'ZA'] and category[:3] == u'A61'
        to_add = int(publication_date[-4:]) >= 2000 and application_number[:2] == u'DE' and category[:3] == u'A61'
        if to_add:
            # if it does fulfill the requirements, fetch its cited patents again (recursion)
            patents_that_will_be_fetched.append(patent_id)
            for patent in cited_patents:
                if patent in apa_ids:
                    continue
                if fetch_cited_patent(patent):
                    cited_patents_to_add.append(patent)
            # write it into a csv file
            dump_to_csv(patent_id, [title, category, publication_number, application_number, publication_date, abstract, description, claims, cited_patents_to_add, pub_years, apa_ids])
            fetched_patents.append(patent_id)
            #print title
            if len(fetched_patents)%50 == 0:
                path_to_logfile='/home/lhelmers/patent_search/patentcrawler/patentcrawler_de.log'
                f = open(path_to_logfile, 'w')
                f.write('fetched_patents = ' + repr(fetched_patents) + '\n')
                f.close()
    # return info to calling ("parent") patent
    print 'ending recursion for %s' %patent_id
    return to_add

def dump_to_csv(id_, entries, dir_=u'/home/lhelmers/patent_search/patentcrawler/german_patent_data/'):
    # entries = [title, category, publication_number, application_number, publication_date, abstract, description, claims, cited_pats_to_add]
    # header = ['title', 'category', 'publication_number', 'application_number', 'publication_date', 'abstract', 'description', 'claims', 'cited_patents']
    df = pd.DataFrame(pd.Series(entries))
    path = dir_ + 'patent_' + id_ + u'.csv'
    df.T.to_csv(path, sep='\t', header=False, index=False, encoding='utf8')

if __name__ == '__main__':
    # start with some seed patent ids
    seed_patents = [u'DE202007017680', u'DE202009010714', u'DE202012100926', u'DE19625253', u'DE102007003515', u'DE20009763']
    # register the ids of already fetched patents
    fetched_patents = []
    visited_patents = []
    patents_that_will_be_fetched = []
    run(seed_patents, fetched_patents, visited_patents, patents_that_will_be_fetched)
