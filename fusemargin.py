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


def get_arguments():
    parser = ArgumentParser(
        description='Fuse two, three or four margins.Please provide test_labels.pkl in the dir')
    #parser.add_argument("split_dir")
    #parser.add_argument("feature_dict", nargs='+',
    #                    help="Can be one or two feature dictionaries")
    #parser.add_argument("--conf_name", default=None,
    #                    help="If defined will save confusions matrices for each split at give output")
    parser.add_argument("--nSplit", type=int)
    parser.add_argument("--nMargin", type=int, default=2)
    parser.add_argument("--m1", default=None)
    parser.add_argument("--m2", default=None)
    parser.add_argument("--m3", default=None)
    parser.add_argument("--m4", default=None)
    parser.add_argument("--test_labels",default='test_labels')
    parser.add_argument("--means",type=str,default=None)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_arguments()
    print "\n"
    print args
    splits_acc=0.0
    if args.means is not None:
        means_list_split = [str(item) for item in args.means.split(';')]    
    for t in range(args.nSplit):
#        import code
#        code.interact(local=locals())
        test_labels = pickle.load(open(args.test_labels+'_split'+str(t)))
        if args.nMargin == 2:          
            file1=args.m1+'_split'+str(t)
            file2=args.m2+'_split'+str(t)
            margin1=pickle.load(open(file1))
            margin2=pickle.load(open(file2))
            if args.means is not None:
		means_list = [float(item) for item in means_list_split[t].split(' ')]
                weight1 = means_list[0]/sum(means_list)
                weight2 = means_list[1]/sum(means_list)
                mean_margin=(weight1*margin1)+(weight2*margin2)                
            else:
                mean_margin=(margin1+margin2)/2
        if args.nMargin == 3:
            file1=args.m1+'_split'+str(t)
            file2=args.m2+'_split'+str(t)
            file3=args.m3+'_split'+str(t)
            margin1=pickle.load(open(file1))
            margin2=pickle.load(open(file2))
            margin3=pickle.load(open(file3))
            if args.means is not None:
		means_list = [float(item) for item in means_list_split[t].split(' ')]
                weight1 = means_list[0]/sum(means_list)
                weight2 = means_list[1]/sum(means_list)
                weight3 = means_list[2]/sum(means_list)
                mean_margin=(weight1*margin1)+(weight2*margin2)+(weight3*margin3)
            else:
                mean_margin=(margin1+margin2+margin3)/3
        if args.nMargin == 4:
            file1=args.m1+'_split'+str(t)
            file2=args.m2+'_split'+str(t)
            file3=args.m3+'_split'+str(t)
            file4=args.m4+'_split'+str(t)
            margin1=pickle.load(open(file1))
            margin2=pickle.load(open(file2))
            margin3=pickle.load(open(file3))
            margin4=pickle.load(open(file4))
            if args.means is not None:
		means_list = [float(item) for item in means_list_split[t].split(' ')]
                weight1 = means_list[0]/sum(means_list)
                weight2 = means_list[1]/sum(means_list)
                weight3 = means_list[2]/sum(means_list)
		weight4 = means_list[3]/sum(means_list)
                mean_margin=(weight1*margin1)+(weight2*margin2)+(weight3*margin3)+(weight4*margin4)
            else:
                mean_margin=(margin1+margin2+margin3+margin4)/4
        SamplesNum=len(mean_margin)
        ClassNum=len(mean_margin[0])
        res = np.zeros(SamplesNum)
        for k in range(SamplesNum):
            res[k]=np.argmax(mean_margin[k])
        #res[k]=[(np.argmax(mean_margin[k])) for k in range(SamplesNum)]
        confusion = confusion_matrix(test_labels, res)
        correct = (res == test_labels).sum()
        print "Margins fusion " + str(args.m1) + " and " + str(args.m2) +" split "+str(t)+" got " + str((100.0 * correct) / test_labels.size) \
                 + "% correct"
        splits_acc=splits_acc+((100.0 * correct) / test_labels.size)
    splits_acc=splits_acc / args.nSplit
    print "mean accuracy: " + str(splits_acc)

