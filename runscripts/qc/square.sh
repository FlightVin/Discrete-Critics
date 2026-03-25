#!/bin/bash

# source /home/mananaga/miniconda/etc/profile.d/conda.sh
# conda activate qc
export MUJOCO_GL=egl
export WANDB_ENTITY=RL-experiments

# # Make pip-installed cuDNN/CUDA libs visible to JAX
# NVIDIA_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib
# export LD_LIBRARY_PATH=$NVIDIA_LIB:$LD_LIBRARY_PATH

cd "$(dirname "$0")/../.."

python main.py \
    --run_group=QC \
    --project_name=BSC \
    --agent.actor_type=best-of-n \
    --agent.actor_num_samples=32 \
    --env_name=square-mh-low_dim \
    --sparse=True \
    --horizon_length=5 \
    --n_eval_envs=32 \
    --eval_episodes=2
