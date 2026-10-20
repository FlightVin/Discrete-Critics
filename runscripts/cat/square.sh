#!/bin/bash
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --job-name=CAT
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_96GB
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

export MUJOCO_GL=egl
export WANDB_ENTITY=RL-experiments
cd /home/mananaga/Discrete-Critics

# OGPO uv env; ml_collections + robosuite 1.5.1 live in .pydeps
PYTHON=/home/mananaga/ogpo/manan-libero-vla-fix/OGPO/.venv/bin/python
export PYTHONPATH="/home/mananaga/Discrete-Critics/.pydeps:${PYTHONPATH:-}"

# === Defaults ===
run_group="CAT"
project_name="BSC"
wandb_name="Square-CAT"  # {Task}-{Method}[-variant]
actor_type="best-of-n"
actor_num_samples=32
env_name="square-mh-low_dim"
sparse=True
horizon_length=5
num_bins=101
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

$PYTHON main.py \
    --agent=agents/accat.py \
    --run_group=$run_group \
    --project_name=$project_name \
    --wandb_name=$wandb_name \
    --agent.actor_type=$actor_type \
    --agent.actor_num_samples=$actor_num_samples \
    --agent.num_bins=$num_bins \
    --env_name=$env_name \
    --sparse=$sparse \
    --horizon_length=$horizon_length \
    --n_eval_envs=$n_eval_envs \
    --eval_episodes=$eval_episodes \
    --eval_interval=$eval_interval
