#!/bin/bash

sb qc/square.sh --wandb_name="qc"
sb bsc/square.sh --wandb_name="bsc_mc_online" --use_mc_target=True --train_critic_offline=False
sb bsc/square.sh --wandb_name="bsc_td_online" --use_mc_target=False --train_critic_offline=False
