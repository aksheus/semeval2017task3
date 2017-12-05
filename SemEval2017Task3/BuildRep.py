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
    
    def __init__(self,dot_product=False):
        # initialize google w2v
        # need to change later: dimensions and pre trained embeddings should be passed to constructor 
        self.embedding_model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\abkma\\anlp\\GoogleNews-vectors-negative300.bin', binary=True)
        self.features = [ 'dimension'+str(z+1) for z in range(600) ]
        self.dot_product = dot_product
        if self.dot_product:
            self.features.append('dot')
        self.features.append('categories')
        #self.missing_training_instances = set()
        self.false_test_instances = set()
        return

    def BuildTrainRep(self,question_comments,truth_table,oversample=False,ovsize=5,out_path='.'):
        """
            Build and output train csv file with concatenated embeddings
            feature1,feature2,.........................,feature600 , label , id
            concat(<question-embedding> , <comment embedding> ),label , Q268_R4_C1 

            format of input, key : (Q268_R4,'question text') : [ ('Q268_R4_C1' , 'comment text') , ....]
        """
        with open(join_path(out_path,'train.csv'),'w',encoding='utf=8') as out:
            out.write(','.join(z for z in self.features))
            out.write('\n')
            for question in question_comments.keys():
                question_embedding  = self.GetSentenceEmbedding(question[-1])
                if question_embedding.__class__ != np.ndarray:
                    continue
                for id,comment in question_comments[question]:
                    comment_embedding = self.GetSentenceEmbedding(comment,id)
                    if comment_embedding.__class__ != np.ndarray:
                        continue
                    qc_embedding = np.concatenate((question_embedding,comment_embedding))
                    if not self.dot_product:
                        out.write(','.join(str(v) for v in qc_embedding)+','+truth_table[id])
                    else:
                        dp = np.dot(question_embedding,comment_embedding)
                        out.write(','.join(str(v) for v in qc_embedding)+','+str(dp)+','+truth_table[id])
                    out.write('\n')
        return

    def GetSentenceEmbedding(self,text,id=None):
        words = word_tokenize(text)
        word_vectors = []
        for word in words:
            try:
                word_vectors.append(self.embedding_model.word_vec(word,use_norm=False))
            except KeyError:
                pass
        if word_vectors!=[]:
            sentence_matrix = np.stack(word_vectors)
            return np.mean(sentence_matrix,axis=0)
        else:
            # 15 missed training instances cuz of oov :/
            # 25 test instances must output an instance as prediction is must
            print('missed instance')
            #if id is not None:
            #    self.missing_training_instances.add(id)
            print(text)
            return False

    def GetOverSampledText(self,text,ovsize):
        words = word_tokenize(text)
        candidates = np.zeros((ovsize,ovsize),dtype=str)
        for x in range(len(words)):
            candidates[x] =np.asarray( [ w[0] for w in self.embedding_model.similar_by_word(words[x],topn=ovsize)])
        ovtext = []
        for row in candidates.T:
            ovtext.append(' '.join(s for s in row))
        return ovtext


    def BuildTestRep(self,question_comments,out_path='.'):
        """
           Build and output test csv file
           feature1,feature2,.........................,feature600 , id
           concat(<question-embedding> , <comment embedding> ) , Q268_R4_C1
        """
        with open(join_path(out_path,'test.csv'),'w',encoding='utf=8') as out:
            out.write(','.join(z for z in self.features))
            out.write('\n')
            for question in question_comments.keys():

                question_embedding  = self.GetSentenceEmbedding(question[-1])

                if question_embedding.__class__ != np.ndarray:
                    question_embedding = np.zeros(300)

                for id,comment in question_comments[question]:
                    comment_embedding = self.GetSentenceEmbedding(comment)

                    if comment_embedding.__class__ != np.ndarray:
                        comment_embedding = np.zeros(300)
                        self.false_test_instances.add(id)

                    qc_embedding = np.concatenate((question_embedding,comment_embedding))
                    if not self.dot_product:
                        out.write(','.join(str(v) for v in qc_embedding)+','+id)
                    else:
                        dp = np.dot(question_embedding,comment_embedding)
                        out.write(','.join(str(v) for v in qc_embedding)+','+str(dp)+','+id)
                    out.write('\n')
        return
        



        
