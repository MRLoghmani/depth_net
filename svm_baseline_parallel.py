# -*- coding: utf-8 -*-
import h5py
from tqdm import tqdm
from sklearn.cross_validation import train_test_split, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, StandardScaler
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
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.pipeline import make_pipeline


class RunParams:
    def __init__(self, args):
        self.pca_dims = args.PCA_dims
        self.kernel_name = args.kernel_name
        self.save_kernel = self.kernel_name is not None
        self.tuneParams = args.tuneParams
        self.C = args.C
        self.normalize = args.normalize
        self.standardize = args.standardize
        self.saveMargin = args.saveMargin
        self.SelectKBest = args.SelectKBest
        self.penalty = args.penalty
        self.skip_svm = args.skip_svm
        self.doKNN = args.doKNN
type_regex = re.compile(ur'_([rgbdepthcrop]+)\.png')

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
    parser.add_argument("--SelectKBest", type=int, default=None)
    parser.add_argument("--tuneParams", action="store_true")
    parser.add_argument("--kernel_name", default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--C", type=float, default=1)
    parser.add_argument("--saveMargin", default=None)
    parser.add_argument("--penalty", default='l2')
    parser.add_argument("--skip_svm", action="store_true")
    parser.add_argument("--doKNN", action="store_true")
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
        print "Running K fold parameter optimization"
        (X, y) = load_split(join(split_dir, args.split_prefix +
                                 'train_split_0.txt'), features, classes)
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        if runParams.standardize:
            sc = StandardScaler(copy=False)
            X = sc.fit_transform(X)
        if runParams.normalize:
            X = normalize(X)
        Cvals = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        penalties = ['l1', 'l2']
        best = (1, 'l2', 0)
        kf = KFold(X.shape[0], 5)
        for p in penalties:
            for c in Cvals:
                score = 0.0
                for train, test in tqdm(kf):
                    dual = X[train].shape[0] < X[train].shape[1]
                    if p == 'l1':
                        dual = False  # dual not supported with l1 penalty
                        c = c * 100  # l1 needs less regularization
                    clf = svm.LinearSVC(dual=dual, C=c, max_iter=50, penalty=p)
                    clf.fit(X[train], y[train])
                    score += clf.score(X[test], y[test])
                score = score / len(kf)
                print "%s mean %f dual: %s - C: %f, score: %f, P: %s" % (str(X[train].shape),
                                                                         X[train].mean(), str(dual), c, score, p)
                if score > best[2]:
                    best = (c, p, score)
        runParams.C = best[0]
        runParams.penalty = best[1]
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
    print "Mean %f std: %f" % (np.mean(splits_acc), np.std(splits_acc))


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
    print "Loaded %s train and %s test - starting svm" % (str(train_features.shape), str(test_features.shape))
    if runParams.standardize:
        scaler = StandardScaler(copy=False)
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
    if runParams.normalize:
        print "Will normalize data"
        train_features = normalize(train_features, copy=False)
        test_features = normalize(test_features, copy=False)
    if runParams.save_kernel:
        print "Saving kernel"
        save_kernel_matrix(train_features, test_features, train_labels,
                           test_labels, runParams.kernel_name + "_" + str(x))
    if runParams.skip_svm:
        return
    splits_acc[x] = do_svm(LoadedData(
        train_features, train_labels, test_features, test_labels), x, runParams)


def save_kernel_matrix(train_data, test_data, train_labels, test_labels, out_name):
    f = h5py.File(out_name, "w")
    trainK = f.create_dataset("train_kernel", (train_data.shape[0], train_data.shape[0]),
                              compression="gzip", compression_opts=9, dtype="float32")
    testK = f.create_dataset("test_kernel", (train_data.shape[0], test_data.shape[0]),
                             compression="gzip", compression_opts=9, dtype="float32")
    trainL = f.create_dataset("train_labels", (train_data.shape[0], ), compression="gzip",
                              compression_opts=9, dtype="uint16")
    testL = f.create_dataset("test_labels", (test_data.shape[0], ), compression="gzip",
                             compression_opts=9, dtype="uint16")

    trainK[:] = train_data.dot(train_data.T)
    trainL[:] = train_labels
    testK[:] = train_data.dot(test_data.T)
    testL[:] = test_labels
    f.close()


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
    if runParams.SelectKBest:
        anova_filter = SelectKBest(f_regression, k=runParams.SelectKBest)
        clf = svm.LinearSVC(dual=False, C=runParams.C)
        anova_svm = make_pipeline(VarianceThreshold(), anova_filter, clf)
        anova_svm.fit(train_data, loaded_data.train_labels)
        res = anova_svm.predict(test_data)
    else:
        dual = train_data.shape[0] < train_data.shape[1]
        if runParams.doKNN is False:
            print "Svm params: C: %f, dual: %s, penalty %s" % (runParams.C, str(dual), runParams.penalty)
            clf = svm.LinearSVC(dual=dual, C=runParams.C, penalty=runParams.penalty)  # C=0.00001 good for JHUIT
        else: 
            print "Running KNN"
            clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_data, loaded_data.train_labels)
        res = clf.predict(test_data)

    if runParams.saveMargin:
	if runParams.SelectKBest:
	    Margins = anova_svm.decision_function(test_data)
        else:
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
    names = get_readable_list(f['datasetNames'], f)
    feats = {}
    for i in range(len(names)):
        feats[names[i]] = features[i][:featSize]
    return feats


def get_features(args):
    print "Loading precomputed features"
    extension = splitext(args.feature_dict[0])[1]
    if extension == '.mat':
        return get_hdf5_feats(args.feature_dict[0])
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
