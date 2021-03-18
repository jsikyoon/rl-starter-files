import os, sys
import subprocess
from datetime import datetime

###############################################################################
# Params
###############################################################################
img_name = 'minigrid:latest'

## gpu
gpu_ids = '2'

## algo
algo = ['ppo']

## env
env = ['MiniGrid-RedBlueDoors-6x6-v0']
#env = ['MiniGrid-MemoryS13Random-v0']

## mem
mem_type = ['trxl', 'trxli', 'gtrxl-gru']
#mem_type = ['lstm']
mem_len = [128]
n_layer = [5]
recurrence = [1]

## etc
save_interval = 10
frames = 1000000

###############################################################################
# Volumn options
###############################################################################
volumn_options = [
        "-v /common:/common -v /data/local/jy651/:/data/local/jy651",
        "-v /cortex/users/jy651:/cortex/users/jy651"
        ]
volumn_options = " ".join(volumn_options) + " "

###############################################################################
# Run
###############################################################################

def run (algo, env, mem_type, mem_len, n_layer, rec):
    if mem_type == 'lstm':
        cont_name = '_'.join([algo, mem_type+'Rec'+str(rec), env])
    elif 'trxl' in mem_type:
        cont_name = '_'.join([algo,
            mem_type+'Memlen'+str(mem_len)+'Nlayer'+str(n_layer)+'Rec'+str(rec),
            env])
    else:
        raise ValueError

    run_command = f'python3 -m scripts.train \
            --algo {algo} \
            --env {env} \
            --recurrence {rec} \
            --mem_type {mem_type} \
            --mem_len {mem_len} \
            --n_layer {n_layer} \
            --save-interval {save_interval} \
            --frames {frames}'

    command = 'docker run -d '
    command += volumn_options
    command += \
        '--device=/dev/nvidiactl --device=/dev/nvidia-uvm --runtime nvidia '
    for _gpu_id in gpu_ids.split(','):
        command += '--device=/dev/nvidia'+_gpu_id+' '
    command += '-e NVIDIA_VISIBLE_DEVICES='+gpu_ids+' '
    command += '--name '+cont_name+' '
    command += img_name
    command += ' /bin/bash -c "'
    command += 'cd /cortex/users/jy651/gym-minigrid && pip install -e . && '
    command += 'cd /cortex/users/jy651/torch-ac && pip install -e . && '
    command += 'cd /cortex/users/jy651/rl-starter-files && '
    command += run_command+'"'
    print(command)
    os.system(command)

for _algo in algo:
    for _env in env:
        for _rec in recurrence:
            for _mem_type in mem_type:
                for _mem_len in mem_len:
                    for _n_layer in n_layer:
                        run(_algo, _env, _mem_type, _mem_len, _n_layer, _rec)

