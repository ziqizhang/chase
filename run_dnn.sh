#!/bin/bash
#export PYTHONPATH=/home/ziqizhang/chase/python/src
export PYTHONPATH=/home/zqz/Work/chase/python/src
input=/home/zqz/Work/chase/data/ml/ml/rm/labeled_data_all.csv
output=/home/zqz/Work/chase/output

SETTINGS=("input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-4,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax"
       "input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-4,maxpooling1d=2,lstm=100-True,gmaxpooling1d,dense=2-softmax")

IFS=""

echo ${#SETTINGS[@]}

for s in ${SETTINGS[*]}
do
    printf '\n'
    echo ">>> Start the following setting at $(date): "
    line="\t${s}"
    echo -e $line
    python3 -m ml.classifier_dnn ${s}
    echo "<<< completed at $(date): "
done



