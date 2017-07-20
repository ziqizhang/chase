#!/bin/bash
#$ -l h_rt=120:00:00
# below -m option can have any combination of b , e or a  to imply when to to send email where;
#    b = begining of job  e = end of job  a = in case job gets aborted unexpectedly 
#$ -m be
#$ -M ziqi.zhang@ntu.ac.uk 
#$ -l mem= 16G
#$ -l rmem= 14G
module load apps/python/anaconda3-2.5.0
source activate myexp
cd chase
./run_svm.sh

