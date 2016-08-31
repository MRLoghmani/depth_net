# -*- coding: utf-8 -*-
import h5py
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
from os.path import join, splitext
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
        self.C = args.C
        self.normalize = args.normalize
        self.saveMargin = args.saveMargin
type_regex = re.compile(ur'_([depthcrop]+)\.png')

LoadedData = namedtuple(
    "LoadedData", "train_patches train_labels test_patches test_labels")


def get_arguments():
    parser = ArgumentParser(
        description='SVM based classification for whole images.')
    parser.add_argument("split_dir")
    parser.add_argument("feature_dict", nargs='+',
                        help="Can be one or two feature dictionaries")
    parser.add_argument("--conf_name", default=None,
                        help="If defined will save confusions matrices for each split at give output")
    parser.add_argument("--split_prefix", default='depth_')
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--mkl_threads", type=int, default=2)
    parser.add_argument("--classes", type=int, default=51)
    parser.add_argument("--PCA_dims", type=int, default=None)
    parser.add_argument("--tuneParams", action="store_true")
    parser.add_argument("--kernel_name", default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--C", type=float, default=1)
    parser.add_argument("--saveMargin", default=None)
    parser.add_argument("--feature_size", default=None, type=int)
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


def prepare_jobs(split_dir, features, n_splits, jobs, classes, runParams):
    if runParams.tuneParams:
        print "Running parameter optimization"
        (X, y) = load_split(join(split_dir, args.split_prefix +
                                 'train_split_0.txt'), features, classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0)
        tuned_parameters = [{'C': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}]
        scores = ['precision', 'recall']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(svm.LinearSVC(dual=False), tuned_parameters, cv=5,
                               scoring='%s_weighted' % score, n_jobs=8)
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean_score, scores.std() * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            runParams.C = clf.best_params_['C']

    jobs_todo = []
    jobs_running = []
    splits_acc = Array('d', range(n_splits))
    for i in range(n_splits):
        jobs_todo.append(Process(target=run_split, name="Split" + str(i),
                                 args=(split_dir, features, n_splits, splits_acc, i, classes, runParams)))
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


def load_split(split_path, feat_dict, classes):
    f_size = feat_dict[feat_dict.keys()[0]].shape[0]
    ft_lines = open(split_path, 'rt').readlines()
    samples = get_samples_per_class(ft_lines, classes)
    features = []
    for c in range(classes):
        features.append(np.empty((samples[c], f_size)))
    ccounter = np.zeros(classes, dtype='int')
    for line in ft_lines:
        [path, classLabel] = line.split()
        nClass = int(classLabel)
        features[nClass][ccounter[nClass]] = feat_dict[path][:f_size]
        ccounter[nClass] += 1
    labels = np.hstack([np.ones(data.shape[0]) * c for c,
                        data in enumerate(features)]).ravel()
    features = np.vstack(features)
    return (features, labels)


def run_split(split_dir, features, n_splits, splits_acc, x, classes, runParams):
    print "Loading split %d" % x
    (train_features, train_labels) = load_split(join(split_dir,
                                                     args.split_prefix + 'train_split_' + str(x) + '.txt'), features, classes)
    (test_features, test_labels) = load_split(join(split_dir,
                                                   args.split_prefix + 'test_split_' + str(x) + '.txt'), features, classes)
    print "Loaded %s train and %s test - starting svm"
    if runParams.normalize:
        start = time.time()
        print "Will normalize data"
        train_features = normalize(train_features, copy=False)
        test_features = normalize(test_features, copy=False)
        print "It took %f seconds" % (time.time() - start)
    if runParams.save_kernel:
        print "Saving kernel"
        save_kernel_matrix(train_features, test_features, train_labels,
                           test_labels, runParams.kernel_name + "_" + str(x))
    splits_acc[x] = do_svm(LoadedData(
        train_features, train_labels, test_features, test_labels), x, runParams)


def save_kernel_matrix(train_data, test_data, train_labels, test_labels, out_name):
    data = {}
    data["train_kernel"] = train_data.dot(train_data.T)
    data["test_kernel"] = train_data.dot(test_data.T)
    data["train_labels"] = train_labels
    data["test_labels"] = test_labels
    io.savemat(out_name, mdict=data)


def do_svm(loaded_data, split_n, runParams):
    #    loaded_data = load_split(args.input_dir, args.train_file, args.test_file)
    print "Fitting SVM to data - train data %s, test data %s" \
        % (str(loaded_data.train_patches.shape), str(loaded_data.test_patches.shape))
    if runParams.pca_dims:
        PCA_dims = runParams.pca_dims
        print "Will perform PCA to reduce dimensions to %d" % PCA_dims
        start = time.time()
        pca = PCA(n_components=PCA_dims)
        pca.fit(loaded_data.train_patches)
        print "PCA computed, now transforming"
        train_data = pca.transform(loaded_data.train_patches)
        test_data = pca.transform(loaded_data.train_patches)
        end = time.time()
        print "It took %f seconds to perform PCA" % (end - start)
        print "Fitting SVM to data - train data %s, test data %s" \
            % (str(train_data.shape), str(test_data.shape))
    else:
        train_data = loaded_data.train_patches
        test_data = loaded_data.test_patches
    print "Feature mean %f and std %f" % (train_data.mean(), train_data.std())
    start = time.time()
    clf = svm.LinearSVC(dual=False, C=runParams.C)  # C=0.00001 good for JHUIT
    clf.fit(train_data, loaded_data.train_labels)
    res = clf.predict(test_data)
    if runParams.saveMargin:
        Margins = clf.decision_function(test_data)
        filemargins = open(runParams.saveMargin+'_split'+str(split_n), 'w')
        ftest_labels = open('test_labels'+'_split'+str(split_n), 'w') #enable only if loaded_data.test_labels change
        pickle.dump(loaded_data.test_labels, ftest_labels)
        pickle.dump(Margins,filemargins)
        filemargins.close()
    confusion = confusion_matrix(loaded_data.test_labels, res)
    if conf_path is not None:
        np.savetxt(conf_path + '_' + str(split_n) + '.csv', confusion)
    correct = (res == loaded_data.test_labels).sum()
    end = time.time()
    print "Split " + str(split_n) + " Got " + str((100.0 * correct) / loaded_data.test_labels.size) \
        + "% correct, took " + str(end - start) + " seconds "
    return (100.0 * correct) / loaded_data.test_labels.size


def get_readable_list(name, f):
    readable = []
    for x in range(name.size):
        obj = f[f[name[0][x]][0][0]]
        readable.append(''.join(chr(i) for i in obj[:]))
    return readable


def get_hdf5_feats(path, featSize=None):
    import ipdb; ipdb.set_trace()
    print "Loading hdf5\mat file"
    f = h5py.File(path)
    features = f['X'][:].T
    if featSize is None:
        featSize = features.shape[1]
    names = get_readable_list(f['datasetNames'][:], f)
    feats = {}
    for i in range(len(names)):
        feats[names[i]] = features[i][:featSize]
    return feats


def get_features(args):
    print "Loading precomputed features"
    extension = splitext(args.feature_dict[0])[1]
    if extension == '.mat':
        return get_hdf5_feats(args.feature_dict[0], args.feature_size)
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
    firstline = open(join(args.split_dir, args.split_prefix +
                          'train_split_0.txt'), 'rt').readlines()[0]
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
#    import ipdb; ipdb.set_trace()
    start_time = time.time()
    args = get_arguments()
    mkl.set_num_threads(args.mkl_threads)
    print "\n"
    print args
    conf_path = args.conf_name
    features = get_features(args)
    if features is None:
        print "Features not found or corruped - exiting"
        quit()
    params = RunParams(args)
    prepare_jobs(args.split_dir, features, args.splits,
                 args.jobs, args.classes, params)
    elapsed_time = time.time() - start_time
    print " Total elapsed time: %d " % elapsed_time
