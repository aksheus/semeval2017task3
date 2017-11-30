"""

 USAGE: class which reads xml format and relevancy file

"""
from bs4 import BeautifulSoup as bs

class PreProcess:

    def __init__(self,path):
        self.path = path

    def GetQuestionCommentDict():
        """  return dict of this format
             key : ('Q268','question text') : [ ('Q268_R4_C1' , 'comment text') , ....]
        """
        pass

    def GetTruthTable():
        """
           From relevancy file build this table
           key : 'Q268_R4_C1' : true/false
        """
        pass
