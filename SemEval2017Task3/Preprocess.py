"""

 USAGE: class which reads xml format and relevancy file

"""
from bs4 import BeautifulSoup as bs

class PreProcess:

    def __init__(self):
        return

    def GetQuestionCommentDict(self,path):
        """  return dict of this format
             key : 'question text' : [ ('Q268_R4_C1' , 'comment text') , ....]
        """
        qc = {}
        soup = None
        with open(path,errors='ignore') as inp:
            soup = bs(inp,'lxml-xml')
        questions = soup.find_all('RelQuestion')
        comments = soup.find_all('RelComment')
        for question in questions:
            relevant_comments = [ (z['RELC_ID'],z.text.rstrip().strip('\n')) for z in comments 
                                 if question['RELQ_ID'] in z['RELC_ID']]
            qc[question.text.rstrip().strip('\n')] = relevant_comments
        return qc


    def GetTruthTable(self,path):
        """
           From relevancy file build this table
           key : 'Q268_R4_C1' : true/false
        """
        truth_table  = {}
        with open(path,errors='ignore') as inp:
            for line in inp:
                pieces = line.rstrip().split()
                truth_table[pieces[1]]=pieces[-1]
        return truth_table
