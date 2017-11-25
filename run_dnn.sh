#!/bin/bash
#export PYTHONPATH=/home/ziqizhang/chase/python/src
export PYTHONPATH=/home/zqz/Work/chase/python/src
input=/home/zqz/Work/chase/data/ml/ml/rm/labeled_data_all.csv
output=/home/zqz/Work/chase/output

#SETTINGS=("input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5,maxpooling1d=5,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-6,maxpooling1d=6,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-6,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax")



#SETTINGS=("input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5,6](dropout=0.2,conv1d=100,maxpooling1d=v),lstm=200-True,gmaxpooling1d,dense=100,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5,6](dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=100,dense=2-softmax" )

#SETTINGS=("input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5](dropout=0.2,conv1d=100,maxpooling1d=v),flatten,dense=500,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5](dropout=0.2,conv1d=100,maxpooling1d=v),flatten,dense=1000,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5](dropout=0.2,conv1d=100,maxpooling1d=v),flatten,dense=1000,dense=200,dense=2-softmax")

#SETTINGS=("input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),lstm=200-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4,lstm=100-True,gmaxpooling1d),dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4,lstm=200-True,gmaxpooling1d),dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-3,maxpooling1d=4,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5,maxpooling1d=4,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),flatten,dense=100,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),conv1d=100-4,lstm=200-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),conv1d=100-4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),conv1d=100-4,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5](dropout=0.2,conv1d=100,maxpooling1d=4),conv1d=100-4,flatten,dense=100,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-3,maxpooling1d=4,conv1d=50-3,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-4,maxpooling1d=4,conv1d=50-4,flatten,dense=2-softmax" 
#SETTINGS=("input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5,maxpooling1d=4,conv1d=50-5,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-3,maxpooling1d=4,conv1d=50-3,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-4,maxpooling1d=4,conv1d=50-4,flatten,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5,maxpooling1d=4,conv1d=50-5,flatten,dense=2-softmax" )


#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-4-2,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-5-2,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=dropout=0.2,conv1d=100-6-2,maxpooling1d=4,lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2]{1}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2]{1,2}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3]{1}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3]{1,2}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4]{1}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4]{1,2}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"

#SETTINGS=("input=$input output=$output dataset=rm model_desc=b_sub_conv[2]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2]<3>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax")
SETTINGS=("input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[3],so),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[4],so),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[5],so),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[3,4,5]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[3,4,5],so),lstm=100-True,gmaxpooling1d,dense=2-softmax" )
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[2,3,4]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[5]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[5]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[5]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax")
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5]<1,2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5]<1>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[3,4,5]<2>(dropout=0.2,conv1d=100,maxpooling1d=4),lstm=100-True,gmaxpooling1d,dense=2-softmax" g
#"input=$input output=$output dataset=rm model_desc=b_sub_conv[4,5]{2}(dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax" )

#model_desc="b sub_conv[2,3,4](dropout=0.2,conv1d=100,maxpooling1d=v),lstm=100-True,gmaxpooling1d,dense=2-softmax"



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



