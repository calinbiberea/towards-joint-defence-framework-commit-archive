#!/bin/bash

model=$1
dataset=$2
train_flag=$3

for adv in FGSM DeepFool CWL2 PGD100
do
    python3 ./ADV_Generate_LID_Mahalanobis.py --dataset $dataset --net_type $model --adv_type $adv --train $train_flag
done