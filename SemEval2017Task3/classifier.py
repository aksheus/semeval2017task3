import argparse
from Preprocess import PreProcess
import BuildRep
import pandas as pd 
import numpy as np 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,classification_report,make_scorer
from sklearn.ensemble import RandomForestClassifier as RFC,AdaBoostClassifier as Ada
from sklearn.model_selection import cross_validate

"""
      
USAGE: classifier.py --train <path> --test <path> --mode <rep or classify> --out <path/outfilename.pred>

"""

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='USAGE: classifier.py --train <path> --test <path> --mode <rep or classify>')
	parser.add_argument('-tr','--train',required=True)
	parser.add_argument('-te','--test',required=True)
	parser.add_argument('-m','--mode',required=True)
	parser.add_argument('-o','--out',required=False)
	args= vars(parser.parse_args())

	truth_path = r'C:\Users\abkma\anlp\ass3\train.relevancy'

	if args['mode'] == 'rep':
		preprocessor = PreProcess()
		question_comments = preprocessor.GetQuestionCommentDict(args['train'])
		#for k in question_comments.keys():
		#	print(k,' : ',question_comments[k])
		#print(len(question_comments))
		truth_table = preprocessor.GetTruthTable(truth_path)
		for k in truth_table.keys():
			print(k,' : ',truth_table[k])
		print(len(truth_table))

	elif args['mode'] == 'classify':
		pass
	# will 