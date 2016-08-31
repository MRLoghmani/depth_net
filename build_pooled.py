# -*- coding: utf-8 -*-
import tqdm
import numpy as np
import time
import h5py
import pickle
from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser(
        description='This tool turns list of batches into a pooled representation')
    parser.add_argument("filelist_path", help="File containing list of files to parse")
    parser.add_argument("outpath", help="Where to save the dictionary")
    parser.add_argument("mode", help="How to pool the features", default='vanilla')
    return parser.parse_args()


def vanilla_pool(hfile):
    return (hfile['feats'][0], hfile.attrs['relative_path'])


def spatial_pyramid_pooling(hfile, levels=1):
    allFeats = hfile['feats'][:]
    wholeImage = allFeats[0]
    f = allFeats.max(axis=0)
#    for l in range(levels):
        
    return (np.hstack((wholeImage, f)), hfile.attrs['relative_path'])

modeMapping = {'vanilla': vanilla_pool, 'spm': spatial_pyramid_pooling}


def do_pooling(filelist, pool_mode, outname):
    pool_func = modeMapping[pool_mode]
    dic = {}
    for line in tqdm.tqdm(filelist):
        hfile = h5py.File(line.strip())
        (feat, name) = pool_func(hfile)
        dic[name] = feat
    with open(outname, 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
        

if __name__ == '__main__':
    start = time.time()
    args = get_arguments()
    do_pooling(open(args.filelist_path).readlines(), args.mode, args.outpath)
    print "It took %f" % (time.time() - start)
