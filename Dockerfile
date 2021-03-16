#from pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
from pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt-get update && apt-get install -y \
    curl \
    zip \
    unzip \
    git

#######################Jsik option##########################
RUN apt-get install -y \
    vim \
    net-tools \
    openssh-server \
    screen

# Setting
RUN git clone https://github.com/jsikyoon/my_ubuntu_settings --branch indent2 && \
    cd my_ubuntu_settings && \
    ./setup.sh && \
    rm -rf /my_ubuntu_settings
############################################################

# tensorboardX
RUN pip install tensorboardX

# GYM-Minigrid
RUN git clone https://github.com/jsikyoon/gym-minigrid && \
    cd gym-minigrid && \
    pip install -e .

# Torch-ac
RUN git clone https://github.com/jsikyoon/torch-ac && \
    cd torch-ac && \
    pip install -e .

# rl-starter-files
RUN git clone https://github.com/lcswillems/rl-starter-files

WORKDIR rl-starter-files
