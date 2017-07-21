#!/bin/sh
#$ -l h_rt=120:00:00 -m bea -M ziqi.zhang@ntu.ac.uk -l mem=16G -l rmem=14G
module load apps/python/anaconda3-2.5.0
source activate myexp
cd chase
./run_svm.sh

