#!/bin/bash
#SBATCH --job-name Test
#SBATCH --account OPEN-27-67
#SBATCH --partition qgpu
#SBATCH --time 2:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

WORK_DIR=/mnt/proj3/open-24-5/pengjy_new/ICASSP_SSL_SSP_T

source /mnt/proj3/open-24-5/pengjy_new/Support/miniconda/bin/activate /mnt/proj3/open-24-5/pengjy_new/Support/miniconda/envs/py39

cd $WORK_DIR

ml CUDA
ml cuDNN
ml mkl 

name=Baseline.yaml
name_lm=Baseline_lm.yaml

python3 trainSpeakerNet.py --config yaml/$name --distributed >> log/$name.log
python3 trainSpeakerNet.py --config yaml/$name_lm --distributed >> log/$name_lm.log
python3 trainSpeakerNet_Eval.py --config yaml/$name_lm  --eval >> log/$name_lm.log
