MODEL_FOLDER=$1
ls /mnt/BackupB/digits/jobs/${1}/*.caffemodel -t1 | xargs -I '{}' ./do_jhuit_tests.sh '{}' ${2} 2>&1 | tee jhuit_svm.out
