#!/bin/bash
export MUJOCO_GL=egl
export WANDB_ENTITY=RL-experiments
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
    --eval_episodes=2 \
    --eval_interval=100000
