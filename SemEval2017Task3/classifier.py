import argparse
import Preprocess
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

    if args['mode'] == 'rep':
        pass
    elif args['mode'] == 'classify':
        pass
    # will 