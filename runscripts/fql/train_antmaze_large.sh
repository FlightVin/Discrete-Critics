#!/bin/bash

source /home/mananaga/miniconda/etc/profile.d/conda.sh
conda activate floqcritic
export MUJOCO_GL=egl

cd /home/mananaga/BSC_branches/main/Binary-Success-Critic

python main.py \
    --env_name=antmaze-large-navigate-singletask-v0 \
    --agent=agents/fql.py \
    --agent.alpha=10 \
    --save_dir=./exp
