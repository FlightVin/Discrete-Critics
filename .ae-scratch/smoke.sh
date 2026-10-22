cd /home/vineethb/ae/work/v1rec
METHOD=${1:-bsc}
D=.ae-scratch/smoke_$METHOD
python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" $D
T0=$(date +%s)
env MUJOCO_GL=egl ROBOMIMIC_DATA_DIR=/data/user_data/vineethb/robomimic XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  AE_PREEMPT_FLAG=$PWD/.ae-scratch/smoke_flag_never \
  /data/user_data/vineethb/envs/discrete-critics/bin/python train_v1.py --mode=online --method=$METHOD \
  --env_name=square-mh-low_dim --horizon_length=5 --n_eval_envs=2 --eval_episodes=1 --log_interval=250 \
  --ckpt_minutes=0.1 --seed=0 --warmup_steps=2000 --pretrain_updates=500 --online_steps=6000 --eval_interval=2000 \
  --buffer_cap=90000 --init_actor=.ae-scratch/competent_actor.pkl --run_dir=$D > $D.log 2>&1
echo "rc=$? wall=$(( $(date +%s) - T0 ))s"
grep -E "^(PROGRESS|PHASE|EVAL |CKPT_DONE|CONFIG|RUN_|Traceback)" $D.log | cut -c1-220 | head -70
tail -5 $D.log
