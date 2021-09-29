import os
import gym
import gym_minigrid

# For unity
import threading
import portpicker
import zipfile
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper


class UnityGym3D:
  LOCK = threading.Lock()

  def __init__(self, env_file, id=0, action_size=3, size=(64, 64), seed=0):
    assert size[0] == size[1]
    
    with self.LOCK:
      env = UnityEnvironment(file_name=env_file, worker_id=id+seed, seed=seed, timeout_wait=100)
      env = UnityToGymWrapper(env, True,  allow_multiple_obs=True)
    self._env = env
    self._size = size
    self.action_size = action_size

  @property
  def observation_space(self):
    shape = self._size +tuple([3])
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return gym.spaces.Discrete(self.action_size)

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      obs = self._env.reset()
      image = obs[0]
      top_view = obs[2]

    #image = np.transpose(image, (2, 0, 1))  # 3, 64, 64
    #top_view = np.transpose(top_view, (2, 0, 1))  # 3, 64, 64

    #return {'image': image, 'top_view': top_view}
    return [image]

  def step(self, action):
    action = action + 1 # unity reserve action 0 as staying still
    obs, r, done, info = self._env.step(action)

    if r > 0.:
      reward = 3.
    else:
      reward = 0.

    image = obs[0]
    top_view = obs[2]

    #image = np.transpose(image, (2, 0, 1))  # 3, 64, 64
    #top_view = np.transpose(top_view, (2, 0, 1))  # 3, 64, 64
    #obs = {'image': image, 'top_view': top_view}
    obs = [image]

    return obs, reward, done, info


def make_env(unity_env, env_key, seed=None):
    if unity_env:
        unity_env_dir = 'unity_envs'
        os.makedirs(unity_env_dir, exist_ok=True)
        env_name = env_key.split('-')[1]
        if env_name == 'GridWorld':
            env_name = 'GridWorld'
        elif env_name == 'OrderSeq4BallsSparse':
            env_name = 'AreaLSizeL4BallFixPosDist3FgNoResetPos'
        elif env_name == 'OrderSeq4BallsBase':
            env_name = 'AreaLSizeL4BallFixPosDist2FgNoResetPos'
        elif env_name == 'OrderSeq5BallsBase':
            env_name = 'AreaLSizeL5BallFixPosDist2FgNoResetPos'
        else:
            raise NotImplementedError(f'{env_name} is not built yet.')
        if not os.path.exists(os.path.join(unity_env_dir, env_name)):
            if env_name == 'GridWorld':
                os.system(f'wget https://www.dropbox.com/s/gh8z8f0z90f4nvq/GridWorld.zip -P {unity_env_dir}')
                zipfile_name = 'GridWorld.zip'
            elif env_name == 'AreaLSizeL4BallFixPosDist3FgNoResetPos':
                os.system(f'wget https://www.dropbox.com/s/c4u378cptced57m/AreaLSizeL4BallFixPosDist3FgNoResetPos.zip -P {unity_env_dir}')
                zipfile_name = 'AreaLSizeL4BallFixPosDist3FgNoResetPos.zip'
            elif env_name == 'AreaLSizeL4BallFixPosDist2FgNoResetPos':
                os.system(f'wget https://www.dropbox.com/s/sqi9ir7bnkle9ff/AreaLSizeL4BallFixPosDist2FgNoResetPos.zip -P {unity_env_dir}')
                zipfile_name = 'AreaLSizeL4BallFixPosDist2FgNoResetPos.zip'
            elif env_name == 'AreaLSizeL5BallFixPosDist2FgNoResetPos':
                os.system(f'wget https://www.dropbox.com/s/9wzo4eu4qmhucnm/AreaLSizeL5BallFixPosDist2FgNoResetPos.zip -P {unity_env_dir}')
                zipfile_name = 'AreaLSizeL5BallFixPosDist2FgNoResetPos.zip'
            else:
                raise NotImplementedError(f'{env_name} is not built yet.')
            with zipfile.ZipFile(os.path.join(unity_env_dir,zipfile_name),"r") as zip_ref:
                zip_ref.extractall(unity_env_dir)
            dir_name = zipfile_name.split('.')[0]
            os.system(f'chmod 755 -R {unity_env_dir}/{dir_name}')
        if env_name == 'GridWorld':
            env_name = 'GridWorld/GridWorld-linux.x86_64'
            unity_env = UnityEnvironment(os.path.join('unity_envs',env_name),
                base_port=portpicker.pick_unused_port(), side_channels=[])
            env = UnityToGymWrapper(unity_env, True, True, True, seed)
        elif 'AreaLSize' in env_name:
            env = UnityGym3D(os.path.join('unity_envs',env_name,'OrderSeqLinux1Area.x86_64'), seed=seed)
        else:
            raise NotImplementedError(f'{env_name} is not built yet.')
    else:
        env = gym.make(env_key)
        env.seed(seed)
    return env
