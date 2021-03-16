import os, sys
import subprocess
from datetime import datetime

gpu_ids = sys.argv[1]
img_name = sys.argv[2]
cont_name = sys.argv[3]
run_command = ' '.join(sys.argv[4:])

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

