#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=2
#SBATCH --job-name=test
#SBATCH --mail-user=up276@nyu.edu
#SBATCH --output=../../runs/slurm_%j.out

module purge
module load python3/intel/3.5.3
#module load scikit-learn/intel/0.18.1
python3 -c "import sklearn; print(sklearn.__version__)"
module load tensorflow/python3.5/1.0.1
cd /scratch/up276/QueryBasedSummarization/code
python3 -u embed_and_train_baseline_sub_mult_300K.py

