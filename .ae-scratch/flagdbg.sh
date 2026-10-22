cd /home/vineethb/ae/work/v1rec
PY=/data/user_data/vineethb/envs/discrete-critics/bin/python
i=0
for a in "--agent=agents/acbsc.py --mode=online --agent.actor_num_samples=4" "--mode=online --method=bsc --agent=agents/acbsc.py --agent.actor_num_samples=4" "--mode=online --method=bsc --agent.actor_num_samples=4"; do
  i=$((i+1))
  echo "== $a"
  env WANDB_MODE=disabled JAX_PLATFORMS=cpu $PY train_v1.py $a --run_dir=.ae-scratch/r$i --init_actor=/x 2>&1 | grep -E "FATAL|CONFIG|Error" | head -3
done
