"""
This is ascript for extracting some statistics
on the patents contained in the DB
"""
import pickle
import numpy as np
from database2.make_patent_db import load_session, Patent
from nltk.tokenize import word_tokenize

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.types import DateTime
from sqlalchemy.interfaces import PoolListener
import os
def load_session():
    metadata = Base.metadata
    Session = sessionmaker(bind=engine)
    session = Session()
    return session
class MyListener(PoolListener):
    def connect(self, dbapi_con, con_record):
        dbapi_con.execute("PRAGMA temp_store = 2")

#DB_URL = 'sqlite:///' + os.path.join(os.path.dirname(__file__), 'patent.db')
DB_URL = 'sqlite:////media/lea/toshiba/patent.db'
engine = create_engine(DB_URL, listeners=[MyListener()])
Base = declarative_base()

session = load_session()
#patent_sample = session.query(Patent)

def calc_full_corpus_stats():
    #print "Calculating no of citations!"
    n_citations = np.array([len(pat.to_dict()['cited_patents']) for pat in session.query(Patent)])
    #print "Calculating no of words" 
    n_words = np.array([len(word_tokenize(pat.title + ' ' + pat.abstract + ' ' + pat.claims + ' ' + pat.description)) for pat in patent_sample])
    #print "No of citations:" 
    #print "Average: %f" %np.mean(n_citations) 
    #print "Std: %f" %np.std(n_citations) 
    #print "\n" 
    #print "No of words:" 
    #print "Average: %f" %np.mean(n_words) 
    #print "Std: %f" %np.std(n_words) 
    with open('n_citations.pkl', 'wb') as ncit_file:
        pickle.dump(n_citations, ncit_file)
    with open('n_words.pkl', 'wb') as nword_file:
        pickle.dump(n_words, nword_file)
      
        
def load_corpus():
    corpus = np.load('../corpus/corpus.npy', encoding='latin1').item()
    return corpus#[pat.lower().split() for pat in corpus.values()]

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
    

target_ids = np.load('../corpus/target_ids.npy')
n_citations = np.array([len(session.query(Patent).filter(Patent.id == pat_id)[0].to_dict()['cited_patents']) for pat_id in target_ids])
np.mean(np.array(n_citations))
np.std(np.array(n_citations))
#if __name__ == "__main__":
#    calc_full_corpus_stats()