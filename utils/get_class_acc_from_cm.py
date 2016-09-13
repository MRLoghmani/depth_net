# -*- coding: utf-8 -*-
import numpy as np
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("confusion_root")
    parser.add_argument("n_splits", type=int)
    parser.add_argument("--outpath", default=None)
    return parser.parse_args()


def merge_cm(cm_root, splits):
    globalCM = np.loadtxt(cm_root + '0')
    for x in range(1, splits):
        globalCM += np.loadtxt(cm_root + str(x))
    return globalCM

if __name__ == '__main__':
    args = get_args()
    cm = merge_cm(args.confusion_root, args.n_splits)
    acc_class = cm.diagonal()/cm.sum(axis=1)
    if args.outpath:
        outpath = args.outpath
    else:
        outpath = "accuracy_per_class.txt"
        np.savetxt(open(outpath, 'w'), acc_class)
