# -*- coding: utf-8 -*-
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

type_regex = re.compile(ur'_([depthcrop]+)\.png')

LoadedData = namedtuple(
    "LoadedData", "train_patches train_labels test_patches test_labels")


def get_arguments():
    parser = ArgumentParser(
        description='SVM based classification for whole images.')
    parser.add_argument("split_dir")
    parser.add_argument("feature_dict", nargs='+', help="Can be one or two feature dictionaries")
    parser.add_argument("--conf_name", default=None, help="If defined will save confusions matrices for each split at give output")
    parser.add_argument("--split_prefix", default='depth_')
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--mkl_threads", type=int, default=2)
    parser.add_argument("--classes", type=int, default=51)
    args = parser.parse_args()
    return args


def get_samples_per_class(split_lines, n_classes):
    ''' Returns a vector containing the number of samples per class '''
    samples = np.zeros(n_classes, dtype='int')
    for line in split_lines:
        [_, classLabel] = line.split()
        samples[int(classLabel)] += 1
    return samples


def is_alive(job):
    status = job.is_alive()
    if status is False:  # this is important!
        job.join()
        print "Finished job " + job.name
    return status


def prepare_jobs(split_dir, features, n_splits, jobs, classes):
    jobs_todo = []
    jobs_running = []
    splits_acc = Array('d', range(n_splits))
    for i in range(n_splits):
        jobs_todo.append(Process(target=run_split, name="Split"+str(i), args=(split_dir,features,n_splits, splits_acc, i, classes)))
    jobs_todo.reverse()  # just to get the jobs in expected order
    while len(jobs_running) + len(jobs_todo):  # while there are still jobs running or to run 
        if len(jobs_todo) and len(jobs_running) < jobs:
            print "Starting new job"
            new_job = jobs_todo.pop()
            new_job.start()
            jobs_running.append(new_job)
        jobs_running[:] = [j for j in jobs_running if is_alive(j)]
        time.sleep(0.3)
                
    print splits_acc[:]
    print np.mean(splits_acc)

def run_split(split_dir, features, n_splits, splits_acc, x, classes):
#    import ipdb; ipdb.set_trace()
    f_size = features[features.keys()[0]].shape[0]
    print "Loading split %d" % x
    train_lines = open(join(split_dir, args.split_prefix + 'train_split_' + str(x) + '.txt'), 'rt').readlines()
    test_lines = open(join(split_dir, args.split_prefix + 'test_split_' + str(x) + '.txt'), 'rt').readlines()
    # pre allocate space for features
    training_samples = get_samples_per_class(train_lines, classes)
    testing_samples = get_samples_per_class(test_lines, classes)
    train_features = []
    test_features = []
    for c in range(classes):
        train_features.append(np.empty((training_samples[c], f_size)))
        test_features.append(np.empty((testing_samples[c], f_size)))
    # load the features
    ccounter = np.zeros(classes, dtype='int')
    for line in train_lines:
        [path, classLabel] = line.split()
        nClass = int(classLabel)
        train_features[nClass][ccounter[nClass]] = features[path]
        ccounter[nClass] += 1
    ccounter = np.zeros(classes, dtype='int')
    for line in test_lines:
        [path, classLabel] = line.split()
        nClass = int(classLabel)
        test_features[nClass][ccounter[nClass]] = features[path]
        ccounter[nClass] += 1
    train_labels = np.hstack([np.ones(data.shape[0]) * c for c, data in enumerate(train_features)]).ravel()
    test_labels = np.hstack([np.ones(data.shape[0]) * c for c, data in enumerate(test_features)]).ravel()
    test_features = np.vstack(test_features)
    train_features = np.vstack(train_features)
    print "Loaded %s train and %s test - starting svm"
    splits_acc[x] = do_svm(LoadedData(train_features, train_labels, test_features, test_labels), x)

def do_svm(loaded_data, split_n):
    #    loaded_data = load_split(args.input_dir, args.train_file, args.test_file)
    print "Fitting SVM to data - train data %s, test data %s" \
        % (str(loaded_data.train_patches.shape), str(loaded_data.test_patches.shape))
    clf = svm.LinearSVC(dual=False)
    print "Feature mean %f and std %f" % (loaded_data.train_patches.mean(), loaded_data.train_patches.std())
    start = time.clock()
    clf.fit(loaded_data.train_patches, loaded_data.train_labels)
    res = clf.predict(loaded_data.test_patches)
    confusion = confusion_matrix( loaded_data.test_labels, res)
    if conf_path is not None:
        np.savetxt(conf_path + '_' + str(split_n) + '.csv', confusion)
    correct = (res == loaded_data.test_labels).sum()
    score = clf.score(loaded_data.test_patches, loaded_data.test_labels)
    end = time.clock()
    print "Split " + str(split_n) + " Got " + str((100.0 * correct) / loaded_data.test_labels.size) \
        + "% correct, took " + str(end - start) + " seconds " + str(score)
    return score

def get_features(args):
    print "Loading precomputed features"
    feats = args.feature_dict
    if len(feats) > 1:
        return fuse_features(args)
    try:
        with open(feats[0], 'rb') as f:
            return pickle.load(f)
    except:
        return None

def get_type_from_string(path):
    return re.search(type_regex, path).group(1)

def get_split_type(args):
    firstline = open(join(args.split_dir, args.split_prefix + 'train_split_0.txt'), 'rt').readlines()[0]
    path = firstline.split()[0].strip()
    return get_type_from_string(path)

def fuse_features(args):
    feats = args.feature_dict
    print "Fusing features"
    with open(feats[0], 'rb') as f:
        first = pickle.load(f)
    with open(feats[1], 'rb') as f:
        second = pickle.load(f)
    second_type = get_type_from_string(second.keys()[0])
    first_type = get_type_from_string(first.keys()[0])
    split_type = get_split_type(args)
    needs_switch = first_type != second_type
    feat_dict = {}
    for path in first.keys():
        path2 = path
        if needs_switch:
            path2 = path.replace(first_type, second_type)
        save_path = path.replace(first_type, split_type)
        feat_dict[save_path] = np.hstack([first[path], second[path2]])
    print "Done"
    return feat_dict

if __name__ == '__main__':
    start_time = time.time()
    args = get_arguments()
    mkl.set_num_threads(args.mkl_threads)
    print "\n"
    print args
    #import ipdb; ipdb.set_trace()
    conf_path = args.conf_name
    features = get_features(args)
    if features is None:
        print "Features not found or corruped - exiting"
        quit()
    prepare_jobs(args.split_dir, features, args.splits, args.jobs, args.classes)
    elapsed_time = time.time() - start_time
    print " Total elapsed time: %d " % elapsed_time
