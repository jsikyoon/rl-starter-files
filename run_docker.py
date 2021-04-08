import os, sys
import subprocess
from datetime import datetime

###############################################################################
# Params
###############################################################################
img_name = 'minigrid:latest'

## gpu
gpu_ids = ['3','4','5']
#gpu_ids = ['3','4','5','6','7']
#gpu_ids = ['6','7','1','3','4','5']
#gpu_ids = ['4', '1', '3']
cnt = 0
max_cnt = len(gpu_ids)

## algo
algo = 'ppo'
#algo = 'a2c'

## env
#env = 'MiniGrid-RedBlueDoors-6x6-v0'
#env = 'MiniGrid-MemoryS13Random-v0'
#env = 'MiniGrid-MemoryS11-v0'
env = 'MiniGrid-MemoryS7-v0'
#env = 'MiniGrid-Empty-5x5-v0'
#env = 'MiniGrid-GoodObject-16x16-v0'

## mem
#mem_type = ['trxl', 'trxli', 'gtrxl-gru']
#recurrence = [1] # transformer
mem_type = ['lstm']
#recurrence = [4, 8, 16, 32, 64] # lstm
recurrence = [4] # lstm
mem_len = 128
n_layer = 2

## dreamer
#loss_type = ['rep-agent']
loss_type = ['rep-img']

## etc
save_interval = 10
frames = 50000000

model = None
#model = 'MiniGrid-Empty-5x5-v0_Dreamer_ppo_rep-agent_CombFalse_lstm_Rec4_Lr0.0001_LrRep0.001_LrImg0.0001_Nimg4_Imgdreamer_FPP128_seed1_21-04-06-07-30-49'
#model = 'MiniGrid-Empty-5x5-v0_Dreamer_ppo_rep-img_CombFalse_lstm_Rec4_Lr0.0001_LrRep0.001_LrImg0.0001_Nimg4_Imgimgdreamer_FPP128_seed1_21-04-06-08-49-35'
#model = 'MiniGrid-GoodObject-16x16-v0_Dreamer_ppo_rep-agent_CombFalse_lstm_Rec16_Lr0.0001_LrRep0.001_LrImg0.0001_Nimg16_Imgimgdreamer_FPP32_seed1_21-04-06-10-30-44'
#model = 'MiniGrid-GoodObject-16x16-v0_Dreamer_ppo_rep-img_CombFalse_lstm_Rec16_Lr0.0001_LrRep0.001_LrImg0.0001_Nimg16_Imgimgdreamer_FPP32_seed1_21-04-06-10-28-24'

lr_rep = 0.001
lr_img = 0.0001
combine_loss = [0]
#img_method = ['ppo', 'a2c', 'dreamer']
#img_method = ['ppo', 'a2c']
#img_method = ['dreamer']
#img_method = ['ppo', 'dreamer', 'imgdreamer']
#img_method = ['a2c', 'dreamer', 'imgdreamer']
img_method = ['imgdreamer']
n_imagines = [4]

lr = 0.0001 # lstm

if algo == 'ppo':
    frames_per_proc = 128 # ppo
elif algo == 'a2c':
    frames_per_proc = 8 # a2c
else:
    raise ValueError

#frames_per_proc = 32 # a2c
frames_per_proc = 16 # a2c

visualize = 0
episodes = 1
gif = model

if model is not None:
    visualize = 1

if visualize == 1:
    mem_type = [mem_type[0]]
    recurrence = [recurrence[0]]
    loss_type = [loss_type[0]]
    combine_loss = [combine_loss[0]]
    img_method = [img_method[0]]

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

def run (algo, env, mem_type, rec, combine_loss, img_method, loss_type, n_imagine):

    global gpu_ids, cnt, max_cnt

    date = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    if mem_type == 'lstm':
        cont_name = '_'.join(['Dreamer-'+algo, loss_type, mem_type+'Rec'+str(rec),
            'Lr'+str(lr), 'Comb'+str(combine_loss), 'Img'+str(img_method), 'Nimg'+str(n_imagine), env, date])
    elif 'trxl' in mem_type:
        cont_name = '_'.join(['Dreamer-'+algo, loss_type,
            mem_type+'Memlen'+str(mem_len)+'Nlayer'+str(n_layer)+'Rec'+str(rec),
            'Lr'+str(lr), 'Comb'+str(combine_loss), 'Img'+str(img_method), 'Nimg'+str(n_imagine),
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
            --combine_loss {combine_loss} \
            --n_imagine {n_imagine} \
            --img_method {img_method} \
            --visualize {visualize} \
            --episodes {episodes} \
            --gif {gif}'

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
            for _img_method in img_method:
                for _loss_type in loss_type:
                    for _n_imagines in n_imagines:
                        run(algo, env, _mem_type, _rec, _combine_loss, _img_method, _loss_type, _n_imagines)

