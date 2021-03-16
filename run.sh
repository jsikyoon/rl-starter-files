GPU_IDS='2'
IMAGE='minigrid:latest'
CONTAINER='PPO_LSTM4_RedBlueDoors6x6'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-6x6-v0 \
        --model PPO_LSTM4_RedBlueDoors6x6 \
        --recurrence 4 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='3'
IMAGE='minigrid:latest'
CONTAINER='PPO_LSTM4_RedBlueDoors8x8'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-8x8-v0 \
        --model PPO_LSTM4_RedBlueDoors8x8 \
        --recurrence 4 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='4'
IMAGE='minigrid:latest'
CONTAINER='PPO_LSTM6_RedBlueDoors8x8'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-8x8-v0 \
        --model PPO_LSTM4_RedBlueDoors8x8 \
        --recurrence 6 \
        --save-interval 10 \
        --frames 1000000

