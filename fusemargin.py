from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import re
import pickle
from sklearn.metrics import confusion_matrix
from sklearn import svm
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from os.path import join
import time
from multiprocessing import Process, Array
import mkl
from sklearn.decomposition import PCA
from scipy import io

class RunParams:

    def __init__(self, args):
        self.pca_dims = args.PCA_dims
        self.kernel_name = args.kernel_name
        self.save_kernel = self.kernel_name is not None
        self.tuneParams = args.tuneParams
        self.C = 1
        self.normalize = args.normalize
        self.saveMargin  = args.saveMargin

LoadedData = namedtuple(
    "LoadedData", "train_patches train_labels test_patches test_labels")

def get_arguments():
    parser = ArgumentParser(
        description='Fuse two, three or four margins.')
    #parser.add_argument("split_dir")
    #parser.add_argument("feature_dict", nargs='+',
    #                    help="Can be one or two feature dictionaries")
    #parser.add_argument("--conf_name", default=None,
    #                    help="If defined will save confusions matrices for each split at give output")
    parser.add_argument("--nMargin", type=int, default=2)
    parser.add_argument("--m1", default=None)
    parser.add_argument("--m2", default=None)
    parser.add_argument("--m3", default=None)
    parser.add_argument("--m4", default=None)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_arguments()
    print "\n"
    print args
    test_labels = pickle.load(open('test_labels.pkl'))
    if args.nMargin == 2:
        margin1=pickle.load(open(args.m1))
        margin2=pickle.load(open(args.m2))
        mean_margin=(margin1+margin2)/2
    if args.nMargin == 3:
        margin1=pickle.load(open(args.m1))
        margin2=pickle.load(open(args.m2))
        margin3=pickle.load(open(args.m3))
        mean_margin= (margin1+margin2+margin3)/3
    if args.nMargin == 4:
        margin1=pickle.load(open(args.m1))
        margin2=pickle.load(open(args.m2))
        margin3=pickle.load(open(args.m3))
        margin4=pickle.load(open(args.m4))
        mean_margin= (margin1+margin2+margin3+margin4)/4
        
    SamplesNum=len(mean_margin)
    ClassNum=len(mean_margin[0])
    res = np.zeros(SamplesNum)
    for k in range(SamplesNum):
        res[k]=np.argmax(mean_margin[k])
    #res[k]=[(np.argmax(mean_margin[k])) for k in range(SamplesNum)]
    
#            print k
#            res[k] = [(i if mean_margin[k][p] > 0 else j) for p,(i,j) in enumerate((i,j) 
#                                                    for i in range(ClassNum)
#                                                    for j in range(i+1,ClassNum))]
    confusion = confusion_matrix(test_labels, res)
    correct = (res == test_labels).sum()
    print "Margins fusion  " + str(args.m1) + " and " + str(args.m2) +" got " + str((100.0 * correct) / test_labels.size) \
            + "% correct"

