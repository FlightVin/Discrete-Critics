#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=BSC
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

export MUJOCO_GL=egl
export WANDB_ENTITY=RL-experiments
cd "$(dirname "$0")/../.."
python main.py \
    --run_group=BSC \
    --project_name=BSC \
    --agent=agents/acbsc.py \
    --agent.actor_type=best-of-n \
    --agent.critic_target_bon=False \
    --agent.actor_num_samples=32 \
    --env_name=square-mh-low_dim \
    --sparse=True \
    --horizon_length=5 \
    --n_eval_envs=32 \
    --eval_episodes=2 \
    --eval_interval=100000
