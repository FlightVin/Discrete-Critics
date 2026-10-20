#!/bin/bash

# wandb names: {Task}-{Method} for default configs, {Task}-{Method}-{variant} otherwise.
# Defaults come from each script; plot_eval_success.py reads the exact {Task}-{Method} runs.

# sb qc/square.sh
# sb bsc/square.sh --wandb_name="Square-BSC-mc-online" --use_mc_target=True --train_critic_offline=False
# sb bsc/square.sh --wandb_name="Square-BSC-online" --use_mc_target=False --train_critic_offline=False
sb bsc/square.sh --wandb_name="Square-BSC-contrastive" --use_mc_target=False --contrastive_loss=True
sb bsc/square.sh --wandb_name="Square-BSC-mc-contrastive" --use_mc_target=True --contrastive_loss=True

sb qc/toolhang.sh
sb bsc/toolhang.sh --use_mc_target=False
sb bsc/toolhang.sh --wandb_name="Toolhang-BSC-mc" --use_mc_target=True

# sb arb/square.sh
# sb arb/square.sh --wandb_name="Square-ARB-expected" --arb_readout=expected
# sb arb/toolhang.sh

sb cat/square.sh
sb cat/toolhang.sh
