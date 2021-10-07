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

# For minigrid
from PIL import Image
from gym_minigrid.minigrid import Grid


class UnityGym3D:
  LOCK = threading.Lock()

  def __init__(self, env_file, id=0, action_size=3, size=(64, 64), seed=0):
    assert size[0] == size[1]
    
    with self.LOCK:
      env = UnityEnvironment(file_name=env_file, worker_id=id+seed, seed=seed, timeout_wait=100,
          base_port=portpicker.pick_unused_port())
      env = UnityToGymWrapper(env, True,  allow_multiple_obs=True)
    self._env = env
    self._size = size 
    self.action_size = action_size

  @property
  def observation_space(self):
    shape = self._size +tuple([3])
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return gym.spaces.Discrete(self.action_size)

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      _obs = self._env.reset()
      obs = {}
      obs['image'] = _obs[0]/255.0
    return obs

  def step(self, action):
    action = action + 1 # unity reserve action 0 as staying still
    _obs, reward, done, info = self._env.step(action)
    obs = {}
    obs['image'] = _obs[0]/255.0
    return obs, reward, done, info


class Minigrid:
  LOCK = threading.Lock()

  def __init__(self, env_key, img_obs=False, action_size=3, size=(64, 64), seed=0):
    assert size[0] == size[1]

    with self.LOCK:
        env = gym.make(env_key)
        env.seed(seed)
    
    self._env = env
    self._img_obs = img_obs
    self.action_size = action_size
    self._size = size

  @property
  def observation_space(self):
    if self._img_obs:
      shape = self._size +tuple([3])
      space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float)
    else:
      shape = (self._env.agent_view_size, self._env.agent_view_size, 3)
      space = gym.spaces.Box(low=0, high=10, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return gym.spaces.Discrete(self.action_size)

  def close(self):
    return self._env.close()

  def _get_agentview(self, img, agent_view_size):
    grid, vis_mask = Grid.decode(img)
    partialview = grid.render(
        8,
        agent_pos=(agent_view_size // 2, agent_view_size - 1),
        agent_dir=3,
        highlight_mask=vis_mask
    )
    partialview = Image.fromarray(partialview)
    partialview = np.asarray(partialview.resize(self._size, Image.BILINEAR), dtype=np.float)/255.0
    return partialview
    
  def reset(self):
    with self.LOCK:
      obs = self._env.reset()
      obs = self._env.gen_obs()
      if self._img_obs:
        obs['image'] = self._get_agentview(obs['image'], self._env.agent_view_size)
    return obs

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    if done:
      obs = self._env.reset()
    obs = self._env.gen_obs()
    if self._img_obs:
      obs['image'] = self._get_agentview(obs['image'], self._env.agent_view_size)
    return obs, reward, done, info


def make_env(env_key, img_obs=False, seed=None):
    if env_key.split('-')[0] == 'Unity':
        unity_env_dir = 'unity_envs'
        os.makedirs(unity_env_dir, exist_ok=True)
        env_name = env_key.split('-')[1]
        if env_name == 'GridWorld':
            env_name = 'GridWorld/GridWorld-linux.x86_64'
            action_size = 4
        elif 'AreaLSize' in env_name:
            env_name += '/OrderSeqLinux1Area.x86_64'
            action_size = 3
        env = UnityGym3D(os.path.join('unity_envs',env_name),
            action_size=action_size, seed=seed)
    else:
        env = Minigrid(env_key, img_obs=img_obs)
    return env
