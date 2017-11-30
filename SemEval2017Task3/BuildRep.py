"""

 USAGE: class which builds csv files for training and testing from word embeddings

"""
import Preprocess
import gensim

class BuildRep:
    
    def __init__(self):
        # initialize google w2v
        # self.embedding_model = 
        return

    def BuildTrainRep(question_comments,truth_table,out_path='.'):
        """
            Build and output train csv file with concatenated embeddings
            feature1,feature2,.........................,feature600 , label , id
            concat(<question-embedding> , <comment embedding> ),label , Q268_R4_C1 

        """
        pass

    def BuildTestRep(question_comments):
        """
           Build and output test csv file
           feature1,feature2,.........................,feature600 , id
           concat(<question-embedding> , <comment embedding> ) , Q268_R4_C1
        """
        pass



        
