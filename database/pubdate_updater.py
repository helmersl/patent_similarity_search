#!/usr/bin/python:
# -*- coding: utf8 -*-
import re
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.types import DateTime

engine = create_engine('sqlite:////home/lhelmers/patent_search/patentcrawler/database/patent.db') #, echo=True)

#engine = create_engine('sqlite:////home/lea/Documents/master_thesis/patent_search/database/patent.db')#, echo=True)
Base = declarative_base()


class Patent(Base):
    __tablename__ = 'patent'
    __table_args__ = {'extend_existing': True}

    id = Column(String, primary_key=True)
    category = Column(String)
    pub_number = Column(String)
    app_number = Column(String)
    pub_date = Column(DateTime)
    title = Column(String)
    abstract = Column(String)
    description = Column(String)
    claims = Column(String)

    def to_dict(self):
        d = {}
        d['id'] = self.id
        d['category'] = self.category
        d['pub_date'] = self.pub_date
        d['title'] = self.title
        d['abstract'] = self.abstract
        d['description'] = self.description
        d['claims'] = self.claims
        d['cited_patents'] = [c.cited_pat for c in self.cited_patents]
        return d

    def __repr__(self):
        return "<Patent(id='%s', category='%s', pub_number='%s', app_number='%s',\
                        pub_date='%s', title='%s', abstract='%s', description='%s',\
                        claims='%s', cited_patents='%s')>" %(self.id, self.category,
                        self.pub_number, self.app_number, str(self.pub_date), self.title,
                        self.abstract, self.description, self.claims, self.cited_patents)

class Citation(Base):
    __tablename__ = 'citation'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    citing_pat = Column(String, ForeignKey(Patent.id))
    cited_pat = Column(String, ForeignKey(Patent.id))

Patent.cited_patents = relationship(Citation, primaryjoin="Patent.id == Citation.citing_pat",
                                    backref=backref('citations', uselist=True, viewonly=True),
                                    viewonly=True)

def load_session():
    metadata = Base.metadata
    print metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def fill_db(session):
    # helper function for parsing dates:
    '''
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
    '''
    Base.metadata.create_all(engine)

    #search_pattern = path + '/' + 'patent_*'
    entries = ['id', 'title', 'category', 'pub_number', 'app_number', 'pub_date', 'abstract',
               'description', 'claims', 'cited_patents', 'pub_dates']
    with open('missing_patents.txt') as fl:
        for line in fl:
            #get patent csv filename 
            patent_file = re.findall('US.*', line)[0]
            pat_df = pd.read_csv('/home/lhelmers/patent_search/patentcrawler/google_patent_data/patent_%s' %patent_file, sep='\t', encoding='utf-8', names=entries)
            pubdate_df = pd.read_csv('/home/lhelmers/patent_search/patentcrawler/pubdates/%s' %patent_file, sep='\t', encoding='utf-8')
                

                #pat = pd.read_csv(_file, sep='\t', names=entries, encoding='utf-8')
            try:
                # Check if patent csv file was read correctly
                assert(pat_df.shape[1]==11)
                patent = Patent(id=pat_df.id[0],
                    category=pat_df.category[0],
                    pub_number=pat_df.pub_number[0],
                    app_number=pat_df.app_number[0],
                    pub_date=pd.to_datetime(pubdate_df.pub_date[0]),
                    title=pat_df.title[0],
                    abstract=pat_df.abstract[0],
                    description=pat_df.description[0],
                    claims=pat_df.claims[0])
                cited_pats = [(re.sub('[\[\]u\' ]', '', st)) for st in pat_df.cited_patents[0].split(',')]
                for c in cited_pats:
                    citation = Citation()
                    citation.citing_pat = patent.id
                    citation.cited_pat = c
                    session.add(citation)
                session.add(patent)
            except IndexError, e:
                print 'could not parse %s' %patent_file
                print 'Error: ' + e.reason
            except AssertionError, e:
                print 'could not parse %s' %patent_file
                print e.reason
            except ValueError, e:
                print 'could not parse %s' %patent_file
        session.commit()

if __name__ == '__main__':
    fill_db(load_session())
