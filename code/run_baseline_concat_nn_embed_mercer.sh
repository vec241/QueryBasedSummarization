#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:k80
#PBS -l walltime=1:00:00
#PBS -l mem=32GB
#PBS -N test
#PBS -M up276@nyu.edu
#PBS -j oe
#PBS -m abe

module purge
module load scikit-learn/intel/0.18.1
module load python3/intel/3.5.1
module load numpy/intel/1.9.2
module load pandas/python3.5.1/intel/0.18.1
module load tensorflow/python3.5.1/20161029
cd /scratch/up276/QueryBasedSummarization/code
python3 embed_and_train.py
