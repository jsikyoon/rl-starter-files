import os, sys, time
import datetime
import gpustat
import subprocess

###############################
# Find idle GPUs
###############################
# GPU cluster info.
#min_required_mem = 10000 # e.g., 20000, 40000
min_required_mem = 10000 # e.g., 20000, 40000
used_mem_threshold = 10
servers = [
    #'welling', # gpus are inavailable
    'jurgen', # 16GB
    'hinton', # 24GB
    'sutton',
    'bengio', # 48GB
    'jordan',
    'rumelhart',
]

# find idle gpus
idle_gpus = []
for server in servers:
  try:
    output = subprocess.check_output(
      f'ssh {server} nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv',
      shell=True).decode('utf-8')
    output = output.split('\n')[1:-1]
    # name, total mem, used mem
    for i, gpu in enumerate(output):
      name, total_mem, used_mem = gpu.split(',')
      total_mem = int(total_mem[:-3])
      used_mem = int(used_mem[:-3])
      if total_mem > min_required_mem: # minimum requirement
        if used_mem < used_mem_threshold:
        #if used_mem < 2000: # 1000 MB
          if (server=='hinton') and (i==4): # not working gpu
            continue
          if (server=='bengio') and (i==5): # not working gpu
            continue
          if (server=='jordan') and (i>=3): # reserved by Gautam
            continue
          idle_gpus.append([server, i])
  except:
    print(f'{server} gpus are inavailable')
#print(idle_gpus)
#print(len(idle_gpus));exit(1)

###############################
# Run experiments
###############################
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

envs = [
    #'MiniGrid-IMazeS13-v0',
    'Unity-AreaLSizeL4BallFixPosDist3FgNoResetPos',
]
session_name = 'mfrl-' + '-'.join(envs) + '-'+date
models = [
    # for minigrid
    {'algo':'ppo', 'mem_type':'lstm', 'recurrence':64, 'lr':0.0001, 'frames':10000000, 'procs': 2, 'img_encode': 0},
    #{'algo':'ppo', 'mem_type':'gtrxl-gru', 'ext_len':64, 'mem_len':128, 'n_layer':2, 'lr':0.00001, 'frames':10000000, 'procs': 2, 'img_encode': 1},
]

settings = []
for env in envs:
  for model in models:
    settings.append([env, model])

# when there is no enough resource
if len(idle_gpus) < len(settings):
  print("There are no GPUs to run the experiments")
  exit(1)

# create tmux session
os.system(f"tmux new-session -s {session_name} -d")

for i, setting in enumerate(settings):
  env, model = setting
  config = f' --env {env}'
  for key in model.keys():
    config += f' --{key} {model[key]}'

  server, gpu_idx = idle_gpus[i]
  os.system(f"tmux split-window -v -p 140 -t {session_name}")
  os.system(f'tmux send-keys -t {session_name}:0.{i+1} "ssh {server}" Enter')
  os.system(f'tmux send-keys -t {session_name}:0.{i+1} "cd ~/rl-starter-files/" Enter')
  os.system(f"""
    tmux send-keys -t {session_name}:0.{i+1} "
      CUDA_VISIBLE_DEVICES={gpu_idx} python3 -m scripts.train {config}
      " Enter
  """)
  time.sleep(1)

os.system(f'tmux send-keys -t {session_name}:0.0 "exit" Enter')
os.system(f"tmux select-layout -t {session_name} even-vertical")
  
