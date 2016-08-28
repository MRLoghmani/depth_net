#!/bin/bash
# This script accepts a caffemodel (and optionally a deploy)
# and runs a few default tests on the washington dataset

# Usage: ./do_jhuit_tests.sh caffemodel [deploy]

FEAT_FOLDER=ex_fcs/ # where extracted features are holded
CAFFE_MODEL=$1
DEPLOY=${2:-deploy_noshift.txt}  # user provided prototxt or use the default no_shift version
LAYER=${3:-'pool5'}
JOB_ID=${4:-default}
BSIZE=1
job_re='jobs\/(.+)\/sna'  # regex to find job id from caffemodel
if [[ $CAFFE_MODEL =~ $job_re ]]; then JOB_ID=${BASH_REMATCH[1]}; fi  # get job id
source activate digits_bvlc
NORM_NAME=${FEAT_FOLDER}CIN-DB_caffe_${JOB_ID}_normalized.pkl
#COLORJ_NAME=${FEAT_FOLDER}jhuit_caffe_${JOB_ID}_colorjet.pkl
ORIG_NAME=${FEAT_FOLDER}CIN-DB_caffe_${JOB_ID}_original.pkl
echo Extracting CIN-DB normalized to $NORM_NAME
python feature_extractor.py ../CIN/CIN-DB_normalized CIN_DB/all_depth.txt $DEPLOY $CAFFE_MODEL $NORM_NAME --gpu_id 1 --batch-size 1 --center_data --layer_name $LAYER

echo Extracting CIN-DB original to $ORIG_NAME
python feature_extractor.py ../CIN/CIN-DB CIN_DB/all_depth.txt $DEPLOY $CAFFE_MODEL $ORIG_NAME --gpu_id 0 --batch-size $BSIZE --center_data --layer_name $LAYER
source deactivate
echo "Running SVM on $NORM_NAME"
SECONDS=0
python svm_baseline.py CIN_DB/ $NORM_NAME --splits 10 --jobs 5 --split_prefix cin_db_depth_
echo "Took $SECONDS seconds"

#SECONDS=0
#echo "Running SVM on $ORIG_NAME"
python svm_baseline.py CIN_DB/ $ORIG_NAME --splits 10 --jobs 5 --split_prefix cin_db_depth_
#echo "Took $SECONDS seconds"
