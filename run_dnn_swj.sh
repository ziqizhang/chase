#!/bin/bash
#export PYTHONPATH=/home/ziqizhang/chase/python/src
export PYTHONPATH=/home/zqz/Work/chase/python/src
input=/home/zqz/Work/chase/data/ml/ml/rm/labeled_data_all.csv
output=/home/zqz/Work/chase/output
emg_model=/home/zqz/Work/data/GoogleNews-vectors-negative300.bin.gz
emg_dim=300
emt_model=/home/zqz/Work/data/Set1_TweetDataWithoutSpam_Word.bin
emt_dim=300
data=rm
targets=2
word_norm=0

#new models proposed in swj. although the descriptor does not show dropout, it is used. see code add_skipped_conv1d_submodel_other_layers
SETTINGS=(
#"input=$input output=$output oov_random=0 dataset=rm model_desc=f_(conv1d=100-[3]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output oov_random=0 dataset=rm model_desc=f_(conv1d=100-[4]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output oov_random=0 dataset=rm model_desc=f_(conv1d=100-[5]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
#"input=$input output=$output oov_random=0 dataset=rm model_desc=f_(conv1d=100-[3,4,5]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output oov_random=0 emb_model=$emg_model emd_dim=$emg_dim dataset=rm model_desc=f_ggl0(conv1d=100-[3]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output oov_random=0 emb_model=$emg_model emb_dim=$emg_dim dataset=rm model_desc=f_ggl0(conv1d=100-[4]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output oov_random=0 emb_model=$emg_model emb_dim=$emg_dim dataset=rm model_desc=f_ggl0(conv1d=100-[5]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output oov_random=0 emb_model=$emg_model emb_dim=$emg_dim dataset=rm model_desc=f_ggl0(conv1d=100-[3,4,5]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output oov_random=0 emb_model=$emt_model emb_dim=$emt_dim dataset=rm model_desc=f_t0(conv1d=100-[3]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output oov_random=0 emb_model=$emt_model emb_dim=$emt_dim dataset=rm model_desc=f_t0(conv1d=100-[4]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output oov_random=0 emb_model=$emt_model emb_dim=$emt_dim dataset=rm model_desc=f_t0(conv1d=100-[5]),lstm=100-True,gmaxpooling1d,dense=2-softmax" 
"input=$input output=$output oov_random=0 emb_model=$emt_model emb_dim=$emt_dim dataset=rm model_desc=f_t0(conv1d=100-[3,4,5]),lstm=100-True,gmaxpooling1d,dense=2-softmax" )
#"input=$input output=$output dataset=rm model_desc=f_(conv1d=100-[3,4,5],so),lstm=100-True,gmaxpooling1d,dense=2-softmax" )
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
c=0
for s in ${SETTINGS[*]}
do
    printf '\n'
    c=$[$c +1]
    echo ">>> Start the following setting at $(date): "
    echo $c
    line="\t${s}"
    echo -e $line
    python3 -m ml.classifier_dnn ${s}
    echo "<<< completed at $(date): "
done



