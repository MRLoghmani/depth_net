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
    imageset = set()
    with open(args.input_file) as filelist:
        k = 0
        print "Starting hashing of images"
        for line in filelist:
            I = Image.open(os.path.join(args.input_folder, line).strip())
            imageset.add(str(imagehash.phash(I)))
            if (k % 1000) == 0:
                print "Parsed item %d" % (k)
            k += 1
    print "There are %d unique files in the %d input images" % (len(imageset),
                                                                k)
