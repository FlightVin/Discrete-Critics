#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=QC
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

export MUJOCO_GL=egl
export WANDB_ENTITY=RL-experiments
cd /home/mananaga/BSC_branches/manan_qc/Binary-Success-Critic

# === Defaults ===
run_group="QC"
project_name="BSC"
wandb_name="run-name"
actor_type="best-of-n"
actor_num_samples=32
env_name="square-mh-low_dim"
sparse=True
horizon_length=5
n_eval_envs=32
eval_episodes=2
eval_interval=100000

# === Parse CLI overrides (--key=value) ===
for arg in "$@"; do
  case $arg in
    --*=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      key="${key#--}"
      eval "$key=\"$value\""
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

python main.py \
    --run_group=$run_group \
    --project_name=$project_name \
    --wandb_name=$wandb_name \
    --agent.actor_type=$actor_type \
    --agent.actor_num_samples=$actor_num_samples \
    --env_name=$env_name \
    --sparse=$sparse \
    --horizon_length=$horizon_length \
    --n_eval_envs=$n_eval_envs \
    --eval_episodes=$eval_episodes \
    --eval_interval=$eval_interval
