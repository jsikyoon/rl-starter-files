import os, sys
import gpustat
import subprocess

###############################
# Find idle GPUs
###############################
# GPU cluster info.
min_required_mem = 0 # e.g., 20000, 40000
servers = [
    #'welling', # gpus are inavailable
    'bengio',
    'jordan',
    'hinton',
    'sutton',
    'jurgen',
    'rumelhart'
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
        if used_mem < 10: # 10 MB
          if (server=='hinton') and (i==4): # not working gpu
            continue
          if (server=='jordan') and (i>=3): # reserved by Gautam
            continue
          idle_gpus.append([server, i])
  except:
    print(f'{server} gpus are inavailable')


###############################
# Run experiments
###############################
session_name = 'mfrl-MiniGrid-GoodObject-Random-Penalty-VisibleBall'

envs = [
    'MiniGrid-Empty-5x5-v0',
    'MiniGrid-MemoryS7-v0',
]
models = [
    {'algo':'ppo', 'mem_type':'lstm', 'recurrence':4, 'lr':0.001, 'img_encode':0},
    {'algo':'ppo', 'mem_type':'lstm', 'recurrence':4, 'lr':0.001, 'img_encode':1},
    {'algo':'ppo', 'mem_type':'trxli', 'mem_len':4, 'n_layer':2, 'recurrence':1, 'lr':0.0001, 'img_encode':0},
    {'algo':'ppo', 'mem_type':'trxli', 'mem_len':4, 'n_layer':2, 'recurrence':1, 'lr':0.0001, 'img_encode':1},
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
      python3 -m scripts.train {config}
      " Enter
  """)

os.system(f"tmux select-layout -t {session_name} tiled")
  
