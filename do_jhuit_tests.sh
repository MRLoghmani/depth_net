#!/bin/bash
# This script accepts a caffemodel (and optionally a deploy)
# and runs a few default tests on the washington dataset

# Usage: ./do_jhuit_tests.sh caffem
echo Working on ${1}
FEAT_FOLDER=ex_fcs/ # where extracted features are holded
CAFFE_MODEL=$1
DEPLOY=${2:-deploy_noshift.txt}  # user provided prototxt or use the default no_shift version
layer=${3:-pool5}
BSIZE=1
job_re='jobs\/(.+)\/sna'  # regex to find job id from caffemodel
if [[ $CAFFE_MODEL =~ $job_re ]]; then JOB_ID=${BASH_REMATCH[1]}; fi  # get job id

NORM_NAME=${FEAT_FOLDER}jhuit_caffe_${JOB_ID}_normalized.pkl
COLORJ_NAME=${FEAT_FOLDER}jhuit_caffe_${JOB_ID}_colorjet.pkl
ORIG_NAME=${FEAT_FOLDER}jhuit_caffe_${JOB_ID}_original.pkl
echo Starting work on $CAFFE_MODEL
echo Extracting JHUIT normalized to $NORM_NAME, layer $layer
#python feature_extractor.py ../JHUIT/JHUIT_normalized/ JHUIT/all_depth.txt $DEPLOY $CAFFE_MODEL $NORM_NAME --center_data --batch-size 1 --layer_name ${layer}

#echo Extracting JHUIT colorjet to $COLORJ_NAME
#python feature_extractor.py ../JHUIT/JHUIT_colorjet/ JHUIT/all_depth.txt $DEPLOY $CAFFE_MODEL $COLORJ_NAME --center_data --batch-size $BSIZE  --layer_name fc6

echo Extracting JHUIT original to $ORIG_NAME
python feature_extractor.py ../JHUIT/JHUIT/ JHUIT/all_depth.txt $DEPLOY $CAFFE_MODEL $ORIG_NAME --center_data --batch-size 1  --layer_name ${layer}

#source activate svm
#source deactivate
echo "Running SVM on $NORM_NAME"
SECONDS=0
#python -u svm_baseline_parallel.py JHUIT/ $NORM_NAME --splits 1 --split_prefix jhuit_depth_ --classes 49 --mkl_threads 12  --normalize --C 10
echo "Took $SECONDS seconds"

#SECONDS=0
#echo "Running SVM on $COLORJ_NAME"
#python -u svm_baseline.py JHUIT/ $COLORJ_NAME --splits 1 --split_prefix jhuit_depth_
#echo "Took $SECONDS seconds"

SECONDS=0
echo "Running SVM on $ORIG_NAME"
python -u svm_baseline_parallel.py JHUIT/ $ORIG_NAME --splits 1 --split_prefix jhuit_depth_ --classes 49  --mkl_threads 12 --normalize --C 10
echo "Took $SECONDS seconds"
