import numpy as np
import lmdb
import argparse
import cv2
import sys
import os
caffe_root = '/home/athena/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",help="This txt file contains the input png images")
    parser.add_argument("root_dir",help="The root folder of the pngs files")
    parser.add_argument("output_folder",help="The folder where to save the lmdb")
    parser.add_argument("bits",help="bits precision, 8 or 16")
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
#map_size = num_files * res * res * 4  # 2 is because we have 16bit data
map_size = 2147483648
env = lmdb.open(args.output_folder, map_size=map_size)

if (args.bits=='8'):
    flags=cv2.IMREAD_GRAYSCALE
elif (args.bits=='16'):
    flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_GRAYSCALE
else: 
    print("choose between 8 or 16 bits as third argument!")
    exit()
with env.begin(write=True) as txn:
    # txn is a Transaction object
    counter=0
    print "num files:%d" % num_files
    root_dir=args.root_dir
    for i in range(num_files):
        if ((i % 1000)==0):           
            print "file: %d" % i
        (im_path, label) = images[i].split()
   # import code
   # code.interact(local=locals()) 
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 1
        datum.height = res
   	datum.width = res
    	fullfilepath=os.path.join(root_dir,im_path)
   	img = cv2.imread(fullfilepath, flags)
    	#import code 
    	#code.interact(local=locals())
    	datum.data = img.tobytes()  # or .tostring() if numpy < 1.9
    	datum.label = int(label)
    	str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
    	txn.put(str_id.encode('ascii'), datum.SerializeToString())
