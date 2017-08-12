#This script is specificaly designed for NYU Prince server, to initiate a call on GPU for particular script. You might would have to change the settigns according to your specific server.

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=50:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=2
#SBATCH --job-name=test
#SBATCH --mail-user=up276@nyu.edu
#SBATCH --output=../../runs/slurm_%j.out

module purge
module load python3/intel/3.5.3
python3 -c "import sklearn; print(sklearn.__version__)"
module load tensorflow/python3.5/1.0.1
cd /scratch/up276/QueryBasedSummarization/code  #PLEASE REPLACE THIS PATH WITH YOUR SCRIPT PATH
python3 -u embed_and_train.py  # ENTER THE SCRIPT NAME WHICH YOU WANT TO RUN
