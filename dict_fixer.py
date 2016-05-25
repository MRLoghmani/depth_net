# -*- coding: utf-8 -*-
import pickle
from argparse import ArgumentParser

def get_arguments():
    parser = ArgumentParser(
        description='Fixer')
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    with open(args.input) as pfile:
        dic1 = pickle.load(pfile)
    for path in dic1.keys():
        data = dic1[path]
        dic1[path] = data.reshape(data.size)
    with open(args.output, 'wb') as ofile:
        pickle.dump(dic1, ofile, pickle.HIGHEST_PROTOCOL)
