import torch

import utils
from model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, preprocess_obss, acmodel, mem_type, model_dir=None, argmax=False, use_memory=True, device=None, num_envs=1):
        self.preprocess_obss = preprocess_obss
        self.acmodel = acmodel
        self.mem_type =mem_type
        self.argmax = argmax
        self.use_memory = use_memory
        self.device = device
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            if self.mem_type == 'lstm':
                self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=self.device)
            else:
                self.memories = torch.zeros(self.num_envs, self.n_layer+1, self.mem_len, self.acmodel.semi_memory_size, device=self.device)

        self.prev_action = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.prev_state = torch.zeros((self.num_envs, self.acmodel.semi_memory_size), device=self.device)

        self.acmodel.eval()
        #if hasattr(self.preprocess_obss, "vocab"):
        #    self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                if self.mem_type == 'lstm':
                    dist, _, self.memories, state, _ = self.acmodel(preprocessed_obss,
                            self.memories,
                            self.prev_action, self.prev_state)
                else:
                    dist, _, memories, state, _ = self.acmodel(preprocessed_obss,
                            self.memories.permute(1,2,0,3),
                            self.prev_action, self.prev_state)
                    self.memories = torch.stack(memories,dim=0).permute(2,0,1,3)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        self.prev_action = actions
        self.prev_state = state

        # reward estimation
        est_rewards = self.acmodel.reward_decoder(torch.cat([state, actions.unsqueeze(-1)], dim=-1))

        return actions.cpu().numpy(), est_rewards.detach().cpu().numpy()[0]

    def get_action(self, obs):
        actions, est_rewards = self.get_actions([obs])
        return actions[0], est_rewards[0]
        #return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
