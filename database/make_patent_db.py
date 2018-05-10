#!/usr/bin/python:
# -*- coding: utf8 -*-
import glob
import re
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.types import DateTime
from sqlalchemy.interfaces import PoolListener

#engine = create_engine('sqlite:////home/lhelmers/patent_search/patentcrawler/database/patent.db') #, echo=True)

class MyListener(PoolListener):
    def connect(self, dbapi_con, con_record):
        dbapi_con.execute("PRAGMA temp_store = 2")


DB_URL = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'patent.db')
engine = create_engine(DB_URL, listeners=[MyListener()])
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
'''
class Cited(Base):
    __tablename__ = 'cited'
    __table_args__ = {'extend_existing': True}

    id = Column(String, ForeignKey(Patent.id), primary_key=True)

Cited.patent = relationship(Patent, primaryjoin="Patent.id == Citation.citing_pat",
                            backref=backref('patent', viewonly=True),
                            viewonly=True)
'''
def load_session():
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

def fill_db(path, session):
    # helper function for parsing dates:
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

    Base.metadata.create_all(engine)

    search_pattern = path + '/' + 'patent_*'
    entries = ['id', 'title', 'category', 'pub_number', 'app_number', 'pub_date', 'abstract',
               'description', 'claims', 'cited_patents', 'pub_dates']
    for _file in glob.glob(search_pattern):
        pat = pd.read_csv(_file, sep='\t', names=entries, encoding='utf-8')
        try:
            # Check if patent csv file was read correctly
            assert(pat.shape[1]==11)
            patent = Patent(id=pat.id[0],
                category=pat.category[0],
                pub_number=pat.pub_number[0],
                app_number=pat.app_number[0],
                pub_date=pd.to_datetime(
                            re.sub('[a-zA-Z\xe4\xc3\xa4]{3,5}', monthrepl, re.sub(' ', '/', re.sub('[.]', '', pat.pub_date[0]))),
                            dayfirst=True),
                title=pat.title[0],
                abstract=pat.abstract[0],
                description=pat.description[0],
                claims=pat.claims[0])
            cited_pats = [(re.sub('[\[\]u\' ]', '', st)) for st in pat.cited_patents[0].split(',')]
            for c in cited_pats:
                citation = Citation()
                citation.citing_pat = patent.id
                citation.cited_pat = c
                session.add(citation)
            session.add(patent)
        except IndexError, e:
            print 'could not parse %s' %_file
            print 'Error: ' + e.reason
            print len(pat_data)
        except AssertionError, e:
            print 'could not parse %s' %_file
            print e
    session.commit()

if __name__ == '__main__':
    fill_db('/home/lhelmers/patent_search/patentcrawler/google_patent_data/', load_session())
