#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=cb3418 # required to send email notifcations
export PATH="/vol/bitbucket/cb3418/myvenv/bin/:$PATH"
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100 # or TERM=xterm

python test.py