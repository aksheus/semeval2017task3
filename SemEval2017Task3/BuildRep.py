"""

 USAGE: class which builds csv files for training and testing from word embeddings

"""
import gensim.models
from nltk import word_tokenize
import numpy as np
import os

listdir = lambda z: os.listdir(z)
isfile = lambda z : os.path.isfile(z)
join_path = lambda v,w : os.path.join(v,w)

class BuildRep:
    
    def __init__(self):
        # initialize google w2v
        # need to change later: dimensions and pre trained embeddings should be passed to constructor 
        self.embedding_model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\abkma\\anlp\\GoogleNews-vectors-negative300.bin', binary=True)
        self.features = [ 'dimension'+str(z+1) for z in range(300) ]
        self.features.append('categories')
        return

    def BuildTrainRep(self,question_comments,truth_table,out_path='.'):
        """
            Build and output train csv file with concatenated embeddings
            feature1,feature2,.........................,feature600 , label , id
            concat(<question-embedding> , <comment embedding> ),label , Q268_R4_C1 

            format of input, key : 'question text' : [ ('Q268_R4_C1' , 'comment text') , ....]
        """
        with open(join_path(out_path,'train.csv'),'w',encoding='utf=8') as out:
            out.write(','.join(z for z in self.features))
            out.write('\n')
            for question in question_comments.keys():
                question_embedding  = self.GetSentenceEmbedding(question)
                for id,comment in question_comments[question]:
                    comment_embedding = self.GetSentenceEmbedding(comment)
                    qc_embedding = np.concatenate((question_embedding,comment_embedding))
                    out.write(','.join(str(v) for v in qc_embedding)+','+truth_table[id])
                    out.write('\n')
                    exit(1)
        return

    def GetSentenceEmbedding(self,text):
        words = word_tokenize(text)
        word_vectors = []
        for word in words:
            try:
                word_vectors.append(self.embedding_model.word_vec(word,use_norm=True))
            except KeyError:
                pass
        sentence_matrix = np.stack(word_vectors)
        return np.mean(sentence_matrix,axis=0)

    def BuildTestRep(self,question_comments,out_path='.'):
        """
           Build and output test csv file
           feature1,feature2,.........................,feature600 , id
           concat(<question-embedding> , <comment embedding> ) , Q268_R4_C1
        """
        pass



        
