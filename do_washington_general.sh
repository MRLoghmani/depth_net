#!/bin/bash
# This script accepts a caffemodel (and optionally a deploy)
# and runs a few default tests on the washington dataset

# Usage: ./do_washington_tests.sh caffemodel [deploy]
echo Working on ${1}
FEAT_FOLDER=ex_fcs/ # where extracted features are holded
CAFFE_MODEL=$1
DEPLOY=${2:-deploy_noshift.txt}  # user provided prototxt or use the default no_shift version
layer=${3:-pool5}
JOB_ID=${5:-default}
N_SPLITS=10
BSIZE=64
job_re='jobs\/(.+)\/sna'  # regex to find job id from caffemodel
if [[ $CAFFE_MODEL =~ $job_re ]]; then JOB_ID=${BASH_REMATCH[1]}; fi  # get job id

NORM_NAME=${FEAT_FOLDER}washington_${JOB_ID}_${layer}.pkl

echo Extracting Washington normalized to $NORM_NAME
python -u feature_extractor.py ${4} Washington/all_depth_clean.txt $DEPLOY $CAFFE_MODEL $NORM_NAME --center_data --gpu_id 0 --batch-size $BSIZE --layer_name ${layer}


echo "Running SVM on $NORM_NAME"
SECONDS=0
python -u svm_baseline_parallel.py Washington/splits/ $NORM_NAME --splits $N_SPLITS --jobs 4 # --kernel_name washington_normalized_kernel_${JOB_ID}
echo "Took $SECONDS seconds"
