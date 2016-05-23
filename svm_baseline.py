# -*- coding: utf-8 -*-
import pickle
from sklearn.metrics import confusion_matrix
from sklearn import svm
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from os.path import join
import time
import feature_handler


LoadedData = namedtuple(
    "LoadedData", "train_patches train_labels test_patches test_labels")


def get_arguments():
    parser = ArgumentParser(
        description='SVM based classification for whole images.')
    parser.add_argument("data_dir")
    parser.add_argument("split_dir")
    parser.add_argument("net_proto")
    parser.add_argument("net_model")
    parser.add_argument("feature_dict")
    parser.add_argument("--mean_pixel", type=float)
    parser.add_argument("--mean_file")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--layer_name", help="Default is FC7", default='fc7')
    parser.add_argument("--use_gpu", default=True, help="If set false, will force CPU inference")
    parser.add_argument("--center_data", type=bool, default=False)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--conf_name", default='confusion.csv')
    parser.add_argument("--second_dict")
    args = parser.parse_args()
    return args


def run_washington_splits(data_dir, split_dir, features, n_splits):
    splits_acc = []
    classes = 51
    f_size = features[features.keys()[0]].shape[0]
    for x in range(n_splits):
        print "Loading split %d" % x
        train_features = [np.empty((0, f_size)) for n in range(classes)]
        test_features = [np.empty((0, f_size)) for n in range(classes)]
        train_file = open(
            join(split_dir, 'depth_train_split_' + str(x) + '.txt'), 'rt')
        test_file = open(
            join(split_dir, 'depth_test_split_' + str(x) + '.txt'), 'rt')
        train_lines = train_file.readlines()
        test_lines = test_file.readlines()
        for line in train_lines:
            [path, classLabel] = line.split()
            nClass = int(classLabel)
            train_features[nClass] = np.vstack([train_features[nClass], features[path].reshape(1, f_size)])
        for line in test_lines:
            [path, classLabel] = line.split()
            nClass = int(classLabel)
            test_features[nClass] = np.vstack([test_features[nClass], features[path].reshape(1, f_size)])
        train_labels = np.hstack(
            [np.ones(data.shape[0]) * c for c, data in enumerate(train_features)]).ravel()
        test_labels = np.hstack(
            [np.ones(data.shape[0]) * c for c, data in enumerate(test_features)]).ravel()
        test_features = np.vstack(test_features)
        train_features = np.vstack(train_features)
        print "Loaded %s train and %s test - starting svm"
        splits_acc.append(
            do_svm(LoadedData(train_features, train_labels, test_features, test_labels), x))
    print splits_acc
    print np.mean(splits_acc)

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
    np.savetxt(conf_path + '_' + str(split_n) + '.csv', confusion)
    correct = (res == loaded_data.test_labels).sum()
    score = clf.score(loaded_data.test_patches, loaded_data.test_labels)
    end = time.clock()
    print "Got " + str((100.0 * correct) / loaded_data.test_labels.size) \
        + "% correct, took " + str(end - start) + " seconds " + str(score)
    return score


def make_features(args):
    f_extractor = feature_handler.FeatureCreator(
        args.net_proto, args.net_model, args.mean_pixel, args.mean_file, args.use_gpu,
         layer_name=args.layer_name)
    f_extractor.batch_size = args.batch_size
    f_extractor.center_data = args.center_data
    f_extractor.set_data_scale(args.scale)
    f_extractor.data_prefix = args.data_dir
    split_dir = args.split_dir
    train_lines = open(join(split_dir, 'depth_train_split_0.txt'), 'rt').readlines()
    test_lines = open(join(split_dir, 'depth_test_split_0.txt'), 'rt').readlines()
    all_files = [join(args.data_dir, line.split()[0]) for line in train_lines] + \
                [join(args.data_dir, line.split()[0]) for line in test_lines]
    # preload all features so that they are handled in batches
    f_extractor.prepare_features(all_files)
    with open(args.feature_dict, 'wb') as f:
        pickle.dump(f_extractor.features, f, pickle.HIGHEST_PROTOCOL)

def get_features(args):
    try:
        with open(args.feature_dict, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def fuse_features(args):
    with open(args.feature_dict, 'rb') as f:
        first = pickle.load(f)
    with open(args.second_dict, 'rb') as f:
        second = pickle.load(f)
    for path in first.keys():
        first[path] = np.vstack(first[path], second[path])
    return first

if __name__ == '__main__':
    start_time = time.time()
    args = get_arguments()
    print args
    import ipdb; ipdb.set_trace()
    conf_path = args.conf_name
    if args.second_dict:
        features = fuse_features(args)
    else:
        features = get_features(args)
        if features is None:
            make_features(args)
            print "Relaunch script to run svm"
            quit()
    run_washington_splits(args.data_dir, args.split_dir, features, args.splits)
    elapsed_time = time.time() - start_time
    print " Total elapsed time: %d " % elapsed_time
