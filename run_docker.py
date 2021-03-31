import os, sys
import subprocess
from datetime import datetime

###############################################################################
# Params
###############################################################################
img_name = 'minigrid:latest'

## gpu
gpu_ids = '5'

## algo
algo = 'ppo'
#algo = 'a2c'

## env
env = ['MiniGrid-RedBlueDoors-6x6-v0']
#env = ['MiniGrid-MemoryS13Random-v0']
#env = ['MiniGrid-MemoryS11-v0']
#env = ['MiniGrid-MemoryS7-v0']

## mem
#mem_type = ['trxl', 'trxli', 'gtrxl-gru']
#recurrence = [1] # transformer
mem_type = ['lstm']
#recurrence = [4, 8, 16, 32, 64] # lstm
recurrence = [4] # lstm
mem_len = [128]
n_layer = [2]

## dreamer
#loss_type = 'agent'
#loss_type = 'rep-agent'
loss_type = 'rep-img'
#loss_type = 'rep-agent-img'

## etc
save_interval = 10
frames = 50000000

model = None
#model = 'MiniGrid-RedBlueDoors-6x6-v0_Dreamer_ppo_rep-agent_lstm_Rec4_Lr0.0001_FPP128_seed1_21-03-30-09-28-18'

lr_rep = 0.001
lr_img = 0.0001
combine_loss = 1

if mem_type[0] == 'lstm':
    #lr = 0.001 # lstm
    lr = 0.0001 # lstm
else:
    lr = 0.0001 # transformer

if algo == 'ppo':
    frames_per_proc = 128 # ppo
elif algo == 'a2c':
    frames_per_proc = 8 # a2c
else:
    raise ValueError

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
    date = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    if mem_type == 'lstm':
        cont_name = '_'.join(['Dreamer-'+algo, loss_type, mem_type+'Rec'+str(rec),
            'Lr'+str(lr), 'FPP'+str(frames_per_proc), 'Frames'+str(frames), env, date])
    elif 'trxl' in mem_type:
        cont_name = '_'.join(['Dreamer-'+algo, loss_type,
            mem_type+'Memlen'+str(mem_len)+'Nlayer'+str(n_layer)+'Rec'+str(rec),
            'Lr'+str(lr), 'FPP'+str(frames_per_proc), 'Frames'+str(frames),
            env, date])
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
            --frames {frames} \
            --lr {lr} \
            --frames-per-proc {frames_per_proc} \
            --loss_type {loss_type} \
            --model {model} \
            --lr_rep {lr_rep} \
            --lr_img {lr_img} \
            --combine_loss {combine_loss}'

    command = 'docker run -d '
    #command += '--rm ' # when testing
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

for _env in env:
    for _rec in recurrence:
        for _mem_type in mem_type:
            for _mem_len in mem_len:
                for _n_layer in n_layer:
                    run(algo, _env, _mem_type, _mem_len, _n_layer, _rec)

