import os
import gym
import gym_minigrid

# For unity
import portpicker
import zipfile
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


def make_env(unity_env, env_key, seed=None):
    if unity_env:
        unity_env_dir = 'unity_envs'
        os.makedirs(unity_env_dir, exist_ok=True)
        env_name = env_key.split('-')[1]
        if not os.path.exists(os.path.join(unity_env_dir, env_name)):
            if env_name == 'GridWorld':
                os.system(f'wget https://www.dropbox.com/s/gh8z8f0z90f4nvq/GridWorld.zip -P {unity_env_dir}')
                with zipfile.ZipFile(os.path.join(unity_env_dir,'GridWorld.zip'),"r") as zip_ref:
                    zip_ref.extractall(unity_env_dir)
                os.system(f'chmod 755 -R {unity_env_dir}/GridWorld')
            else:
                raise NotImplementedError(f'{env_name} is not built yet.')
        if env_name == 'GridWorld':
            env_name = 'GridWorld/GridWorld-linux.x86_64'
        unity_env = UnityEnvironment(os.path.join('unity_envs',env_name),
            base_port=portpicker.pick_unused_port(), side_channels=[])
        env = UnityToGymWrapper(unity_env, True, True, True, seed)
    else:
        env = gym.make(env_key)
        env.seed(seed)
    return env
