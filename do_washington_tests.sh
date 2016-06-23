#!/bin/bash
# This script accepts a caffemodel (and optionally a deploy)
# and runs a few default tests on the washington dataset

# Usage: ./do_washington_tests.sh caffemodel [deploy]

FEAT_FOLDER=ex_fcs/ # where extracted features are holded
CAFFE_MODEL=$1
#DEPLOY=${2:-instance_deploy.prototxt}  # user provided prototxt or use the default no_shift version
DEPLOY=${2:-deploy_noshift.txt}  # user provided prototxt or use the default no_shift version
JOB_ID=${3:-default}
N_SPLITS=10
BSIZE=128
echo Working on $1
job_re='jobs\/(.+)\/sna'  # regex to find job id from caffemodel
if [[ $CAFFE_MODEL =~ $job_re ]]; then JOB_ID=${BASH_REMATCH[1]}; fi  # get job id

NORM_NAME=${FEAT_FOLDER}vandal_${JOB_ID}_normalized.pkl
ORIG_NAME=${FEAT_FOLDER}vandal_${JOB_ID}_original.pkl
source activate new_digits
echo Extracting Washington normalized to $NORM_NAME
python -u feature_extractor.py ../Washington/rgbd-normalized_gray/ Washington/all_depth_clean.txt $DEPLOY $CAFFE_MODEL $NORM_NAME --center_data --batch-size $BSIZE
echo Extracting Washington original to $ORIG_NAME
python -u feature_extractor.py ../Washington/rgbd-original/ Washington/all_depth_clean.txt $DEPLOY $CAFFE_MODEL $ORIG_NAME --center_data --batch-size $BSIZE

source activate svm
echo "Running SVM on $NORM_NAME"
SECONDS=0
python -u svm_baseline_parallel.py Washington/splits/ $NORM_NAME --splits $N_SPLITS --jobs 4
echo "Took $SECONDS seconds"

SECONDS=0
echo "Running SVM on $ORIG_NAME"
python -u svm_baseline_parallel.py Washington/splits/ $ORIG_NAME --splits $N_SPLITS  --jobs 4
echo "Took $SECONDS seconds"
