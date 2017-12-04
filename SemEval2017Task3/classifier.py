import argparse
from Preprocess import PreProcess
from BuildRep import BuildRep
import pandas as pd 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,classification_report,make_scorer
from sklearn.ensemble import RandomForestClassifier as RFC,AdaBoostClassifier as Ada
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier

"""
      
USAGE: classifier.py --train <path> --test <path> --mode <rep or classify> --out <path/outfilename.pred> --classify <cross or test>

"""

def get_data_frame(csv_file):

    df = pd.read_csv(csv_file)
    labels = np.asarray([row[-1] for row in df.values])
    df = df.drop(labels=df.columns[-1],axis=1)
    return df,labels

def post_process(predictions,false_ids):
    """
       Return modified predictions by making mandatory 
       false predictions false

       predictions [Q268_R4_C2,1.49980396096993,true]

    """
    for idx in range(len(predictions)):

        if predictions[idx][0] in false_ids:
            predictions[idx][1] = 0.0
            predictions[idx][2] = False
    
    return predictions

    

def write_submission(predictions,out_path='submission.pred'):
    """
       Output submission.pred 

       Q268 Q268_R4_C2  402 1.49980396096993    true
              ids        R+C      scores           predictions

       Input

       predictions [Q268_R4_C2,1.49980396096993,true]"""

    digit_map = {'1': '01','2':'02','3': '03', '4':'04', '5':'05','6': '06', '7':'07', '8':'08','9':'09','0':'10'}

    #relabel = lambda w : 'true' if w else 'false'

    with open(out_path,'w',encoding='utf-8') as out:

         for Id,score,pred in predictions:
            
            columns = []

            split_id = Id.split('_')

            relabel = 'true' if pred else 'false'

            columns.append(split_id[0])
            columns.append(Id)
            columns.append(split_id[1][1:]+digit_map[split_id[2][-1]])
            columns.append(str(score))
            columns.append(relabel)

            out.write('\t'.join(col for col in columns))
            out.write('\n')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='USAGE: classifier.py --train <path> --test <path> --mode <rep or classify>')
    parser.add_argument('-tr','--train',required=True)
    parser.add_argument('-te','--test',required=True)
    parser.add_argument('-m','--mode',required=True)
    parser.add_argument('-o','--out',required=False)
    parser.add_argument('-c','--classify',required=False)
    args= vars(parser.parse_args())

    truth_path = r'C:\Users\abkma\anlp\ass3\train.relevancy'

    missing_train_instances = set(['Q270_R37_C3', 'Q273_R60_C7', 'Q274_R41_C5', 'Q284_R51_C9', 'Q277_R5_C5', 'Q280_R11_C7', 'Q280_R48_C1', 'Q281_R56_C3', 'Q284_R33_C5', 'Q284_R44_C4', 'Q284_R44_C5', 'Q286_R13_C9', 'Q286_R13_C9', 'Q287_R10_C6', 'Q287_R23_C2'])

    false_test_ids = {'Q303_R39_C1', 'Q289_R26_C9', 'Q293_R3_C4', 'Q305_R56_C9', 'Q301_R40_C4', 'Q296_R54_C4', 'Q300_R41_C1', 'Q290_R11_C8', 'Q301_R40_C5', 'Q316_R31_C1', 'Q317_R12_C9', 'Q288_R3_C6', 'Q298_R20_C7', 'Q303_R20_C9', 'Q294_R13_C9', 'Q301_R70_C3', 'Q314_R22_C10', 'Q305_R30_C3', 'Q316_R2_C1', 'Q314_R18_C7', 'Q308_R22_C8', 'Q291_R34_C10', 'Q307_R3_C6'}

    if args['mode'] == 'rep':
        preprocessor = PreProcess()
        #question_comments = preprocessor.GetQuestionCommentDict(args['train'])
        question_comments_test = preprocessor.GetQuestionCommentDict(args['test'])
        #for k in question_comments.keys():
        #   print(k,' : ',question_comments[k])
        #print(len(question_comments))
        #truth_table = preprocessor.GetTruthTable(truth_path,missing_train_instances)
        """for k in truth_table.keys():
            print(k,' : ',truth_table[k])
        print(len(truth_table))"""
        builder = BuildRep(dot_product=True)
        #builder.BuildTrainRep(question_comments,truth_table)
        #missing_train_instances = builder.missing_training_instances
        builder.BuildTestRep(question_comments_test)
        #false_test_ids = builder.false_test_instances
        #print(false_test_ids)
        #print('##########################')
        #print(len(missing_train_instances))
        #print(missing_train_instances)

    elif args['mode'] == 'classify':

        trdf,labels = get_data_frame(args['train'])
        tedf,ids = get_data_frame(args['test'])

        rf = RFC(n_estimators=50,
        criterion='entropy', 
        max_depth=None, 
        min_samples_split=2, 
        max_leaf_nodes=None,
        class_weight = 'balanced',
        #random_state = 512
        )

        svm_poly = svm.SVC(C=1,kernel='poly',gamma = 'auto',class_weight = 'balanced')
        svm_rbf = svm.SVC(C=25,kernel='rbf',gamma = 'auto',class_weight = 'balanced',probability=True)
        svm_linear =svm.SVC(C=5,kernel='linear',gamma = 'auto',class_weight = 'balanced',probability=True) #svm.LinearSVC(C=5,class_weight = 'balanced')

        nb = NB()

        knnu = KNeighborsClassifier(n_neighbors=5, 
                                   weights='uniform', 
                                   algorithm='auto', 
                                   leaf_size=30,
                                   p=2, 
                                   metric='minkowski', 
                                   metric_params=None,
                                  n_jobs=-1)

        knnd = KNeighborsClassifier(n_neighbors=5, 
                                   weights='distance', 
                                   algorithm='auto', 
                                   leaf_size=30,
                                   p=2, 
                                   metric='minkowski', 
                                   metric_params=None,
                                  n_jobs=-1)

        clfs = [svm_rbf,svm_linear,nb]
        clfs = [svm_rbf,svm_linear,nb,knnu,knnd]
        clf_names = ['svm_rbf','svm_linear','nb','knnu','knnd']

        if args['classify'] == 'cross':

            scoring = ['f1_macro','accuracy','precision_macro','recall_macro']
            for clf in clfs:
                print(clf) 
                scores = cross_validate(clf,trdf,labels,scoring = scoring,cv=10,return_train_score=False)
                for s in scores.keys():
                    print(s)
                    for v in scores[s]:
                        print(v)
                        print ('###############################')
                print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        elif args['classify'] == 'test':

            #clf.fit(trdf,labels)
            # get probabilities for predcitions
            # output false for oov indexes viz test instances which are all zeros
            for clf,name in zip(clfs,clf_names):
                clf.fit(trdf,labels)
                #zs =[ clf.predict(tedf) for clf in clfs]
                z = clf.predict(tedf)
                zlog_probs = clf.predict_proba(tedf) 
                required_index = clf.classes_.tolist().index(True)
                predictions = [[id,proba[required_index],prediction] for id,proba,prediction in zip(ids,zlog_probs,z)]
                predictions = post_process(predictions,false_test_ids)
                write_submission(predictions,args['out']+'_'+name+'.predictions')