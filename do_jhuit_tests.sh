#!/bin/bash
# This script accepts a caffemodel (and optionally a deploy)
# and runs a few default tests on the washington dataset

# Usage: ./do_jhuit_tests.sh caffemodel [deploy]
echo Working on ${1}
FEAT_FOLDER=ex_fcs/ # where extracted features are holded
CAFFE_MODEL=$1
DEPLOY=${2:-deploy_noshift.txt}  # user provided prototxt or use the default no_shift version
JOB_ID=${3:-default}
BSIZE=256
job_re='jobs\/(.+)\/sna'  # regex to find job id from caffemodel
if [[ $CAFFE_MODEL =~ $job_re ]]; then JOB_ID=${BASH_REMATCH[1]}; fi  # get job id

NORM_NAME=${FEAT_FOLDER}jhuit_caffe_${JOB_ID}_normalized.pkl
COLORJ_NAME=${FEAT_FOLDER}jhuit_caffe_${JOB_ID}_colorjet.pkl
ORIG_NAME=${FEAT_FOLDER}jhuit_caffe_${JOB_ID}_original.pkl
echo Starting work on $CAFFE_MODEL
source activate new_digits
echo Extracting JHUIT normalized to $NORM_NAME
python feature_extractor.py ../JHUIT/JHUIT_normalized/ JHUIT/all_depth.txt $DEPLOY $CAFFE_MODEL $NORM_NAME --center_data --batch-size $BSIZE

echo Extracting JHUIT colorjet to $COLORJ_NAME
python feature_extractor.py ../JHUIT/JHUIT_colorjet/ JHUIT/all_depth.txt $DEPLOY $CAFFE_MODEL $COLORJ_NAME --center_data --batch-size $BSIZE

echo Extracting JHUIT original to $ORIG_NAME
python feature_extractor.py ../JHUIT/JHUIT/ JHUIT/all_depth.txt $DEPLOY $CAFFE_MODEL $ORIG_NAME --center_data --batch-size $BSIZE

source activate general
echo "Running SVM on $NORM_NAME"
SECONDS=0
python -u svm_baseline.py JHUIT/ $NORM_NAME --splits 1 --split_prefix jhuit_depth_
echo "Took $SECONDS seconds"

SECONDS=0
echo "Running SVM on $COLORJ_NAME"
python -u svm_baseline.py JHUIT/ $COLORJ_NAME --splits 1 --split_prefix jhuit_depth_
echo "Took $SECONDS seconds"

SECONDS=0
echo "Running SVM on $ORIG_NAME"
python -u svm_baseline.py JHUIT/ $ORIG_NAME --splits 1 --split_prefix jhuit_depth_
echo "Took $SECONDS seconds"
