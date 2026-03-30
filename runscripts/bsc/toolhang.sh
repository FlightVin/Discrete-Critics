#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=BSC
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out

export MUJOCO_GL=egl
export WANDB_ENTITY=RL-experiments
cd /home/mananaga/BSC_branches/manan_qc/Binary-Success-Critic

# === Defaults ===
run_group="BSC"
project_name="BSC"
wandb_name="run-name"
actor_type="best-of-n"
critic_target_bon=False
use_mc_target=False
train_critic_offline=True
actor_num_samples=32
env_name="tool_hang-ph-low_dim"
sparse=True
horizon_length=8
start_training=40000
online_steps=5000000
n_eval_envs=32
eval_episodes=2
eval_interval=100000
contrastive_loss=False
contrastive_loss_scale=1.0

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
    --agent=agents/acbsc.py \
    --agent.actor_type=$actor_type \
    --agent.critic_target_bon=$critic_target_bon \
    --agent.use_mc_target=$use_mc_target \
    --agent.train_critic_offline=$train_critic_offline \
    --agent.actor_num_samples=$actor_num_samples \
    --env_name=$env_name \
    --sparse=$sparse \
    --horizon_length=$horizon_length \
    --start_training=$start_training \
    --online_steps=$online_steps \
    --n_eval_envs=$n_eval_envs \
    --eval_episodes=$eval_episodes \
    --eval_interval=$eval_interval \
    --agent.contrastive_loss=$contrastive_loss \
    --agent.contrastive_loss_scale=$contrastive_loss_scale
