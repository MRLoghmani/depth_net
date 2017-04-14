# -*- coding: utf-8 -*-
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import matplotlib


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("feature_dict")
    parser.add_argument("split_path")
    parser.add_argument("outname")
    parser.add_argument("--n_classes", default=51, type=int)
    parser.add_argument("--precomputed", default=None)
    parser.add_argument("--N", default=1, type=int)
    parser.add_argument("--label_names", default=None)
    return parser.parse_args()


def get_samples_per_class(split_lines, n_classes):
    ''' Returns a vector containing the number of samples per class '''
    samples = np.zeros(n_classes, dtype='int')
    for line in split_lines:
        [_, classLabel] = line.split()
        samples[int(classLabel)] += 1
    return samples


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


def onpick(event, labels, label_names):
    index = event.ind
    print '--------------'
    lbls = labels[index]
    if label_names is None:
        print lbls
    else:
        print [label_names[label.astype('int8')].strip() for label in lbls]


def plot_classes(feats, labels, classes, N, label_names, subset=None):
    figure = plt.figure()
#    idx = labels < (classes/N)
    colors = cm.rainbow(np.linspace(0, 1, len(subset)))
    ax = plt.subplot(111)
    for id in range(len(subset)):
        idx = labels == subset[id]
        ax.scatter(feats[idx, 0], feats[idx, 1], c=colors[id], edgecolors=None, linewidths=0, picker=3, label=label_names[subset[id]])
#    plt.colorbar()
    recs = []
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
    figure.canvas.mpl_connect('pick_event', lambda event: onpick(event, labels[idx], label_names))
    plt.show()

if __name__ == '__main__':
    args = get_arguments()
    matplotlib.rcParams['pdf.fonttype'] = 42
#    selectedCLasses = [13, 6, 46, 14, 28, 25]
    selectedCLasses = [0, 27, 29, 34, 2, 35, 36]
    if args.precomputed:
        print "Loading precomputed"
        (feats, labels) = pickle.load(open(args.precomputed))
    else:
        try:
            dic = pickle.load(open(args.feature_dict))
        except:
            print "Not a valid dictionary, exiting"
        (feats, labels) = load_split(args.split_path, dic, args.n_classes)
        goodSamples = np.sort(np.hstack([np.where(labels==i) for i in selectedCLasses])).ravel()
        labels = labels[goodSamples]
        feats = feats[goodSamples]
        model = TSNE(n_components=2, random_state=0)
        pca = PCA(n_components=64)
        print "Applying PCA %s" % str(feats.shape)
        start = time.time()
        feats = pca.fit_transform(feats)
        print "Will now apply TSNE %s (took %f)" % (str(feats.shape), (time.time()-start))
        start = time.time()
        feats = model.fit_transform(feats)
        print "Will now visualize data %s (took %f)" % (str(feats.shape), (time.time()-start))
        pickle.dump((feats,labels), open(args.outname, "w"), pickle.HIGHEST_PROTOCOL)
    if args.label_names:
        label_names = open(args.label_names).readlines()
    else:
        label_names = None
    plot_classes(feats, labels, args.n_classes, args.N, label_names, selectedCLasses)
