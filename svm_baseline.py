# -*- coding: utf-8 -*-
from sklearn import svm
import numpy as np
from h5py import File as HDF5File
from argparse import ArgumentParser
from glob import glob
from os.path import join, basename
from common import init_logging, get_logger
import time
from collections import namedtuple
import feature_handler


def get_arguments():
    parser = ArgumentParser(
        description='SVM based classification for whole images.')
    parser.add_argument("input_dir")
    parser.add_argument("train_file")
    parser.add_argument("test_file")
    args = parser.parse_args()
    return args


def load_split(input_dir, train_file, test_file):
    f_extractor = feature_handler.FeatureCreator(
        '/home/enoon/libs/caffe/models/bvlc_alexnet/deploy.prototxt',
        '/home/enoon/libs/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel')
    train_data = {}
    for


def do_whole_image_svm(args):
    logger = get_logger()
    loaded_data = load_split(args.input_dir, args.train_file, args.test_file)
    k = 'linear'
    c = 1
    logger.info("Fitting SVM to data with " + k + " kernel and " + str(c) +
                " C val")
    clf = svm.SVC(C=c, kernel=k)
    start = time.clock()
    clf.fit(loaded_data.train_patches, loaded_data.train_labels)
    res = clf.predict(loaded_data.test_patches)
    correct = (res == loaded_data.test_labels).sum()
    score = clf.score(loaded_data.test_patches, loaded_data.test_labels)
    end = time.clock()
    logger.info("Got " + str((100.0 * correct) / loaded_data.test_labels.size)
                + "% correct, took " + str(end - start) + " seconds " + str(
                    score))


if __name__ == '__main__':
    init_logging()
    args = get_arguments()
    do_whole_image_svm(args)
