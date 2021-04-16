import os, sys
import subprocess
from datetime import datetime

###############################################################################
# Params
###############################################################################
img_name = 'minigrid:latest'


## gpu
#gpu_ids = ['5','6','7','1','2','3','4']
#gpu_ids = ['1','2','3','4','5','6','7']
gpu_ids = ['4','5','6','7','0']
#gpu_ids = ['6','7','0']
cnt = 0
max_cnt = len(gpu_ids)


## algo
algo = 'dreamer'
#algo = 'ppo'
#algo = 'a2c'


## env
#env = 'MiniGrid-RedBlueDoors-6x6-v0'
#env = 'MiniGrid-MemoryS13Random-v0'
#env = 'MiniGrid-MemoryS11-v0'
env = 'MiniGrid-MemoryS7-v0'
#env = 'MiniGrid-Empty-5x5-v0'
#env = 'MiniGrid-GoodObject-16x16-v0'


## mem
mem = 'trxl'
#mem = 'lstm'
if mem == 'trxl':
    mem_type = ['trxl', 'trxli', 'gtrxl-gru']
    recurrence = [1] # transformer
    mem_len = 64
    n_layer = 2
else:
    mem_type = ['lstm']
    #recurrence = [4, 8, 16, 32, 64] # lstm
    recurrence = [2] # lstm
    mem_len = -1  # Not used
    n_layer = -1


## loss type
if algo == 'dreamer':
    loss_type = ['rep-img']
else:
    loss_type = ['rep-agent']
    #loss_type = ['agent']


## etc
save_interval = 10
frames = 50000000
model = None

lr = 0.001 # lstm
#lr = 0.0001 # transformer
lr_rep = 0.001
lr_img = 0.0
#lr_img = 0.001

#combine_loss = [0,1]
combine_loss = [0]

n_imagines = [4]


if algo == 'ppo':
    frames_per_proc = 128 # ppo
elif algo == 'a2c':
    frames_per_proc = 8 # a2c
elif algo == 'dreamer':
    frames_per_proc = 128 # dreamer
else:
    raise ValueError

if algo == 'dreamer':
    #use_real = [0, 1]
    use_real = [0]
else:
    use_real = [0] # not used

img_epochs = 1
rep_epochs = 100

visualize = 0
episodes = 1
gif = model

if visualize == 1:
    mem_type = [mem_type[0]]
    recurrence = [recurrence[0]]
    loss_type = [loss_type[0]]
    combine_loss = [combine_loss[0]]

###############################################################################
# Volumn options
###############################################################################
volumn_options = [
        "-v /common/home/jy651/:/common/home/jy651 -v /data/local/jy651/:/data/local/jy651",
        "-v /cortex/users/jy651:/cortex/users/jy651"
        ]
volumn_options = " ".join(volumn_options) + " "

###############################################################################
# Run
###############################################################################

def run (algo, env, mem_type, rec, combine_loss, loss_type, n_imagine, use_real):

    global gpu_ids, cnt, max_cnt

    date = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    if 'trxl' in mem_type:
        mem = True
    else:
        mem = rec > 1

    if mem:
        default_model_name = f"{env}_{algo}_{loss_type}_Comb{combine_loss}_{mem_type}_"

        if mem_type == 'lstm':
            default_model_name += f"Rec{rec}_"
        else:
            default_model_name += f"Nlayer{n_layer}_MemLen{mem_len}_"

        if algo == 'dreamer':
            default_model_name += f"LrRep{lr_rep}_LrImg{lr_img}_Nimg{n_imagine}_UseReal{use_real}_"
            default_model_name += f"ImgEpochs{img_epochs}_RepEpochs{rep_epochs}_"
        else:
            default_model_name += f"Lr{lr}_RepEpochs{rep_epochs}_"

        default_model_name += f"FPP{frames_per_proc}_{date}"

    else:
        default_model_name = f"{env}_{algo}_{date}"

    cont_name = default_model_name


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
            --combine_loss {combine_loss} \
            --n_imagine {n_imagine} \
            --visualize {visualize} \
            --episodes {episodes} \
            --gif {gif} \
            --use_real {use_real} \
            --rep_epochs {rep_epochs} \
            --img_epochs {img_epochs}'

    command = 'docker run -d '
    #command += '--rm ' # when testing
    command += volumn_options
    command += '--device=/dev/nvidiactl --device=/dev/nvidia-uvm --runtime nvidia '
    gpu_id = gpu_ids[cnt]
    cnt += 1
    cnt = cnt % max_cnt
    command += '--device=/dev/nvidia'+gpu_id+' '
    command += '-e NVIDIA_VISIBLE_DEVICES='+gpu_id+' '
    command += '--name '+cont_name+' '
    command += img_name
    command += ' /bin/bash -c "'
    command += 'cd /common/home/jy651/gym-minigrid && pip install -e . && '
    command += 'cd /common/home/jy651/torch-ac && pip install -e . && '
    command += 'cd /common/home/jy651/rl-starter-files && '
    command += 'pip install matplotlib && '
    command += 'pip install array2gif && '
    command += run_command+'"'
    print(command)
    os.system(command)

for _rec in recurrence:
    for _mem_type in mem_type:
        for _combine_loss in combine_loss:
            for _loss_type in loss_type:
                for _n_imagines in n_imagines:
                    for _use_real in use_real:
                        run(algo, env, _mem_type, _rec, _combine_loss, _loss_type, _n_imagines, _use_real)

