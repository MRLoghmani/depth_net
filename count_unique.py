import argparse
from PIL import Image
import imagehash
import os

#from random import randrange


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("input_file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    #    imageset = set()
    goodhash = {}
    simplehash = {}
    with open(args.input_file) as filelist:
        k = 0
        print "Starting hashing of images"
        for line in filelist:
            impath = os.path.join(args.input_folder, line).strip()
            I = Image.open(impath)
            goodhash[str(imagehash.phash(I))] = impath
            simplehash[str(imagehash.phash_simple(I))] = impath
            if (k % 2000) == 0:
                print "Parsed item %d - good %d/%d" % (k, len(goodhash),
                                                       len(simplehash))
            k += 1
    print "There are %d/%d unique files in the %d input images" % (
        len(goodhash), len(simplehash), k)
    with open("hash.txt","wt") as hf:
        for k in goodhash.keys():
            hf.write(goodhash[k]+"\n")
    with open("simple_hash.txt","wt") as hf:
        for k in simplehash.keys():
            hf.write(simplehash[k]+"\n")
