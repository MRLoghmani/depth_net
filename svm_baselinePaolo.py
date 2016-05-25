# -*- coding: utf-8 -*-
from sklearn import svm
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from os.path import join
import time
import feature_handler
from joblib import Parallel
import copy as c
from multiprocessing import Process, Value, Array

LoadedData = namedtuple(
    "LoadedData", "train_patches train_labels test_patches test_labels")


def get_arguments():
    parser = ArgumentParser(
        description='SVM based classification for whole images.')
    parser.add_argument("data_dir")
    parser.add_argument("split_dir")
    parser.add_argument("net_proto")
    parser.add_argument("net_model")
    parser.add_argument("--mean_pixel", type=float)
    parser.add_argument("--mean_file")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--layer_name", help="Default is FC7", default='fc7')
    parser.add_argument("--use-gpu", type=bool, default=True, help="If set false, will force CPU inference")
    parser.add_argument("--center_data", type=bool, default=False)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--splits", type=int, default=10)

    args = parser.parse_args()
    return args

def run_washington_splits(data_dir, split_dir, f_extractor, n_splits):
    splits_acc = Array('d', range(10))
    classes = 51
    f_size = f_extractor.f_size
    preload = None
    x=0
    all_files=[]
    #preparing the features 
    print "Preparing features..." 
    train_features = [np.empty((0, f_size)) for n in range(classes)]
    test_features = [np.empty((0, f_size)) for n in range(classes)]
    train_file = open(
        join(split_dir, 'depth_train_split_' + str(0) + '.txt'), 'rt')
    test_file = open(
        join(split_dir, 'depth_test_split_' + str(0) + '.txt'), 'rt')
    train_lines = train_file.readlines()
    test_lines = test_file.readlines()
    all_files = [join(data_dir, line.split()[0]) for line in train_lines] + \
                [join(data_dir, line.split()[0])
                 for line in test_lines]
    f_extractor.prepare_features(all_files)
    jobs = []
    for i in range(5):
        p1 = Process(target=run_split, args=(data_dir,split_dir,f_extractor,splits_acc,classes,f_size,all_files,i*2))
        p2 = Process(target=run_split, args=(data_dir,split_dir,f_extractor,splits_acc,classes,f_size,all_files,(i*2)+1))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
    splits_acc2=[i*100 for i in splits_acc]
    print splits_acc2[:]
    print np.mean(splits_acc2)

def run_split(data_dir, split_dir, f_extractor,splits_acc,classes,f_size,all_files,x):
    
    print "Loading split %d" % x 
    train_features = [np.empty((0, f_size)) for n in range(classes)]
    test_features = [np.empty((0, f_size)) for n in range(classes)]
    train_file = open(
        join(split_dir, 'depth_train_split_' + str(x) + '.txt'), 'rt')
    test_file = open(
        join(split_dir, 'depth_test_split_' + str(x) + '.txt'), 'rt')
    train_lines = train_file.readlines()
    test_lines = test_file.readlines()
    try:
        for line in train_lines:
            [path, classLabel] = line.split()
            nClass = int(classLabel)
            train_features[nClass] = np.vstack(
                [train_features[nClass], f_extractor.get_features(join(data_dir, path)).reshape(1, f_extractor.f_size)])
    
        for line in test_lines:
            [path, classLabel] = line.split()
            nClass = int(classLabel)
            test_features[nClass] = np.vstack(
                [test_features[nClass], f_extractor.get_features(join(data_dir, path)).reshape(1, f_extractor.f_size)])
    except:
        print "Unexpected error:", sys.exc_info()[0]
    train_labels = np.hstack(
        [np.ones(data.shape[0]) * c for c, data in enumerate(train_features)]).ravel()
    test_labels = np.hstack(
        [np.ones(data.shape[0]) * c for c, data in enumerate(test_features)]).ravel()
    test_features = np.vstack(test_features)
    train_features = np.vstack(train_features)
    print "Loaded %s train and %s test - starting svm"
    splits_acc[x]=do_svm(LoadedData(train_features, train_labels, test_features, test_labels))
    #import pdb; pdb.set_trace()
    return splits_acc

def do_svm(loaded_data):
    #    loaded_data = load_split(args.input_dir, args.train_file, args.test_file)
    print "Fitting SVM to data - train data %s, test data %s" \
        % (str(loaded_data.train_patches.shape), str(loaded_data.test_patches.shape))
    clf = svm.LinearSVC(dual=False)
    print "Feature mean %f and std %f" % (loaded_data.train_patches.mean(), loaded_data.train_patches.std())
    start = time.clock()
    clf.fit(loaded_data.train_patches, loaded_data.train_labels)
    res = clf.predict(loaded_data.test_patches)
    correct = (res == loaded_data.test_labels).sum()
    score = clf.score(loaded_data.test_patches, loaded_data.test_labels)
    end = time.clock()
    print "Got " + str((100.0 * correct) / loaded_data.test_labels.size) \
        + "% correct, took " + str(end - start) + " seconds " + str(score)
    return score


if __name__ == '__main__':
    args = get_arguments()
    f_extractor = feature_handler.FeatureCreator(
        args.net_proto, args.net_model, args.mean_pixel, args.mean_file,
        use_gpu=args.use_gpu, layer_name=args.layer_name)
    f_extractor.batch_size = args.batch_size
    f_extractor.center_data = args.center_data
    f_extractor.set_data_scale(args.scale)
    run_washington_splits(args.data_dir, args.split_dir, f_extractor, args.splits)
