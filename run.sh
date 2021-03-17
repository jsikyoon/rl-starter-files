: << "END"

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
CONTAINER='PPO_LSTM8_RedBlueDoors8x8'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-8x8-v0 \
        --model PPO_LSTM8_RedBlueDoors8x8 \
        --recurrence 8 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='5'
IMAGE='minigrid:latest'
CONTAINER='PPO_LSTM16_RedBlueDoors8x8'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-8x8-v0 \
        --model PPO_LSTM16_RedBlueDoors8x8 \
        --recurrence 16 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='6'
IMAGE='minigrid:latest'
CONTAINER='PPO_LSTM32_RedBlueDoors8x8'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-8x8-v0 \
        --model PPO_LSTM32_RedBlueDoors8x8 \
        --recurrence 32 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='2'
IMAGE='minigrid:latest'
CONTAINER='PPO_LSTM64_RedBlueDoors8x8'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-8x8-v0 \
        --model PPO_LSTM64_RedBlueDoors8x8 \
        --recurrence 64 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='3'
IMAGE='minigrid:latest'
CONTAINER='PPO_LSTM128_RedBlueDoors8x8'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-8x8-v0 \
        --model PPO_LSTM128_RedBlueDoors8x8 \
        --recurrence 128 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='2'
IMAGE='minigrid:latest'
CONTAINER='PPO_TRXLMemlen20Nlayer7_RedBlueDoors6x6'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-6x6-v0 \
        --recurrence 1 \
        --mem_type trxl \
        --mem_len 20 \
        --n_layer 7 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='2'
IMAGE='minigrid:latest'
CONTAINER='PPO_TRXLiMemlen20Nlayer7_RedBlueDoors6x6'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-6x6-v0 \
        --recurrence 1 \
        --mem_type trxli \
        --mem_len 20 \
        --n_layer 7 \
        --save-interval 10 \
        --frames 1000000

END

GPU_IDS='1'
IMAGE='minigrid:latest'
CONTAINER='PPO_GTRXLGRUMemlen20Nlayer7_RedBlueDoors6x6'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-6x6-v0 \
        --recurrence 1 \
        --mem_type gtrxl-gru \
        --mem_len 20 \
        --n_layer 7 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='2'
IMAGE='minigrid:latest'
CONTAINER='PPO_TRXLMemlen128Nlayer2_RedBlueDoors6x6'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-6x6-v0 \
        --recurrence 1 \
        --mem_type trxl \
        --mem_len 128 \
        --n_layer 2 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='3'
IMAGE='minigrid:latest'
CONTAINER='PPO_TRXLiMemlen128Nlayer2_RedBlueDoors6x6'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-6x6-v0 \
        --recurrence 1 \
        --mem_type trxli \
        --mem_len 128 \
        --n_layer 2 \
        --save-interval 10 \
        --frames 1000000

GPU_IDS='4'
IMAGE='minigrid:latest'
CONTAINER='PPO_GTRXLGRUMemlen128Nlayer2_RedBlueDoors6x6'
python docker_run.py $GPU_IDS $IMAGE $CONTAINER \
    python3 -m scripts.train \
        --algo ppo \
        --env MiniGrid-RedBlueDoors-6x6-v0 \
        --recurrence 1 \
        --mem_type gtrxl-gru \
        --mem_len 128 \
        --n_layer 2 \
        --save-interval 10 \
        --frames 1000000


