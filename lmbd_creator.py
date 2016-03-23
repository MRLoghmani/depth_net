import numpy as np
import lmdb
import caffe
import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",help="This file contains the input HDR images")
    parser.add_argument("output_folder",help="The folder where to save the lmdb")

    args = parser.parse_args()
    return args

N = 1000
res = 256
args = parse_args()

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
images = open(args.input_file).readlines()
num_files  = len(images)
map_size = num_files * res * res * 2  # 2 is because we have 16bit data

env = lmdb.open(args.output_folder, map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(num_files):
        (im_path, label) = images[i].split()
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 1
        datum.height = res
        datum.width = res
        img = cv2.imread(im_path, cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE)
        datum.data = img.tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(label)
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
