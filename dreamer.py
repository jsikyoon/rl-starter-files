import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Bernoulli

class Dreamer_encoder_indices(nn.Module):
    def __init__(self, obs_space, action_space, beta_rep_kl=1.0):
        super().__init__()

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.semi_memory_size = self.image_embedding_size
        self.memory_size = 2*self.memory_size

        # Define memory (LSTM)
        memory_inp_size = self.semi_memory_size + 1 # state + action
        self.memory_module = nn.LSTMCell(memory_inp_size, self.semi_memory_size)

        # prior net
        self.prior_net = nn.Sequential(
                nn.Linear(self.semi_memory_size, self.semi_memory_size),
                nn.ELU(),
                nn.Linear(self.semi_memory_size, 2*self.semi_memory_size)
        )

        # posterior net
        self.post_net = nn.Sequential(
                nn.Linear(self.semi_memory_size+self.image_embedding_size, self.semi_memory_size),
                nn.ELU(),
                nn.Linear(self.semi_memory_size, 2*self.semi_memory_size)
        )

        # Define image decoder
        self.image_deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, (3,3)),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, (3,3)),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 20, (3,3)), # color_idx: 6, obj_idx: 11, state_idx: 3
        )

        # Define reward decoder
        self.reward_decoder = nn.Sequential(
                nn.Linear(64+1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
        )

    def forward(self):
        pass

    def _img_encode(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        return x

    def _rssm_update(self, prev_action, prev_state, memory):

        # [s_{t-1}, a_{t-1}]
        mem_inp = torch.cat([prev_action.unsqueeze(-1), prev_state], dim=-1)

        # RSSM update
        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_module(mem_inp, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        return embedding, memory

    def _post_sampling(self, embedding, x):

        # q(s_t | s_{t-1}, a_{t-1}, o_t)
        post_mean, post_std = self.post_net(
                torch.cat([embedding, x], dim=-1)).split(
                (self.semi_memory_size, self.semi_memory_size), dim=-1)
        post_std = F.softplus(post_std) + 0.1
        post_dist = Normal(post_mean, post_std)
        post = post_dist.rsample()

        return post, post_dist


    def state_rep(self, obs, memory, prev_action, prev_state):
        x = self._img_encode(obs)

        embedding, memory = self._rssm_update(prev_action, prev_state, memory)

        rep, _ = self._post_sampling(embedding, x)

        return rep, memory

    def _prior_sampling(self, embedding):

        # p(s_t | s_{t-1}, a_{t-1})
        prior_mean, prior_std = self.prior_net(embedding).split(
                (self.semi_memory_size, self.semi_memory_size), dim=-1)
        prior_std = F.softplus(prior_std) + 0.1
        prior_dist = Normal(prior_mean, prior_std)
        prior = prior_dist.rsample()

        return prior, prior_dist

    def state_gen(self, memory, prev_action, prev_state):

        embedding, memory = self._rssm_update(prev_action, prev_state, memory)

        rep, _ = self._prior_sampling(embedding)

        return rep, memory

    def rep_loss(self, obs, memory, prev_action, prev_state, actions, rewards):

        x = self._img_encode(obs)

        embedding, memory = self._rssm_update(prev_action, prev_state, memory)

        post, post_dist = self._post_sampling(embedding, x)

        prior, prior_dist = self._prior_sampling(embedding)

        recon_loss = self._recon_loss(obs, post)

        reward_loss = self._reward_loss(post, actions, rewards)

        # KL
        kl_loss = kl_divergence(post_dist, prior_dist).mean()

        # representation loss
        rep_loss = {}
        for key in recon_loss.keys():
            rep_loss[key] = recon_loss[key]
        for key in reward_loss.keys():
            rep_loss[key] = recon_loss[key]
        rep_loss['kl_loss'] = kl_loss
        rep_loss['rep_loss'] = rep_loss['recon_loss'] -1*rep_loss['reward_logprob'] + \
                self.beta_rep_kl*kl_loss

        return rep_loss

    def _recon_loss(self, obs, embedding):

        # [bs*7*7,3]
        img = obs.image.transpose(1, 3).transpose(2, 3)
        img = img.permute(0,2,3,1).view(-1, 3).type(torch.LongTensor).to(embedding.device)

        # [bs,20,7,7]
        pred_img = self.image_deconv(embedding.unsqueeze(-1).unsqueeze(-1))

        # [bs*7*7,11], [bs*7*7,6], [bs*7*7,3]
        [pred_obj, pred_col, pred_state] = pred_img.split((11,6,3), dim=1)
        pred_obj = pred_obj.permute(0,2,3,1).reshape(-1, 11)
        pred_col = pred_col.permute(0,2,3,1).reshape(-1, 6)
        pred_state = pred_state.permute(0,2,3,1).reshape(-1, 3)

        # [bs*7*7,1], [bs*7*7,1], [bs*7*7,1]
        _, pred_obj_idx = torch.max(F.softmax(pred_obj, dim=-1), dim=-1)
        pred_obj_idx = pred_obj_idx.unsqueeze(-1)
        _, pred_col_idx = torch.max(F.softmax(pred_col, dim=-1), dim=-1)
        pred_col_idx = pred_col_idx.unsqueeze(-1)
        _, pred_state_idx = torch.max(F.softmax(pred_state, dim=-1), dim=-1)
        pred_state_idx = pred_state_idx.unsqueeze(-1)

        # [bs*7*7, 3]
        pred_idx = torch.cat([pred_obj_idx, pred_col_idx, pred_state_idx], dim=-1)

        # [bs, 7*7*3]
        pred_idx = pred_idx.reshape(obs.image.shape[0], -1)

        # [bs, 7*7*3]
        img_for_acc = img.reshape(obs.image.shape[0], -1)

        # reconstruction accuracy of a single partial observation
        _correctness = pred_idx == img_for_acc
        correctness = _correctness[:,0]
        for i in range(1, _correctness.shape[1]):
            correctness *= _correctness[:,i]
        recon_acc = correctness.sum().item() / obs.image.shape[0]

        # reconstruction loss
        obj_loss = nn.CrossEntropyLoss()(pred_obj, img[:,0])
        col_loss = nn.CrossEntropyLoss()(pred_col, img[:,1])
        state_loss = nn.CrossEntropyLoss()(pred_state, img[:,2])

        recon_loss = obj_loss + col_loss + state_loss

        Recon_loss = {}
        Recon_loss['recon_acc'] = recon_acc
        Recon_loss['recon_loss'] = recon_loss
        Recon_loss['recon_col_loss'] = col_loss
        Recon_loss['recon_obj_loss'] = obj_loss
        Recon_loss['recon_state_loss'] = state_loss

        return Recon_loss

    def _reward_loss(self, embedding, actions, rewards):

        pred_reward = self.reward_decoder(
                torch.cat([embedding, actions.unsqueeze(-1)], dim=-1))

        # nonzero reward loss
        nonzero_rewards = rewards[rewards!=0]
        nonzero_pred_reward = pred_reward[rewards!=0, :]
        nonzero_reward_num = len(nonzero_rewards)
        if nonzero_reward_num == 0:
            nonzero_reward_mse = torch.Tensor([0]).to(rewards.device)
            nonzero_reward_logprob = torch.Tensor([0]).to(rewards.device)
        else:
            nonzero_reward_mse = nn.MSELoss(reduction='sum')(nonzero_pred_reward,
                nonzero_rewards.unsqueeze(-1))
            nonzero_reward_dist = Normal(nonzero_pred_reward, 1.0)
            #nonzero_reward_dist = Bernoulli(logits=nonzero_pred_reward) # for binary reward
            nonzero_reward_logprob = nonzero_reward_dist.log_prob(
                    nonzero_rewards.unsqueeze(-1)).sum()


        # zero reward loss
        zero_rewards = rewards[rewards==0]
        zero_pred_reward = pred_reward[rewards==0, :]
        zero_reward_num = len(zero_rewards)
        if zero_reward_num == 0:
            zero_reward_mse = torch.Tensor([0]).to(rewards.device)
            zero_reward_logprob = torch.Tensor([0]).to(rewards.device)
        else:
            zero_reward_mse = nn.MSELoss(reduction='sum')(zero_pred_reward,
                zero_rewards.unsqueeze(-1))
            zero_reward_dist = Normal(zero_pred_reward, 1.0)
            #zero_reward_dist = Bernoulli(logits=zero_pred_reward) # for binary reward
            zero_reward_logprob = zero_reward_dist.log_prob(zero_rewards.unsqueeze(-1)).sum()

        reward_mse = nn.MSELoss()(pred_reward, rewards.unsqueeze(-1))
        reward_dist = Normal(pred_reward, 1.0)
        #reward_dist = Bernoulli(logits=pred_reward) # for binary reward
        reward_logprob = reward_dist.log_prob(rewards.unsqueeze(-1)).mean()

        Reward_loss = {}

        Reward_loss['reward_mse'] = reward_mse
        Reward_loss['reward_logprob'] = reward_logprob
        Reward_loss['nonzero_reward_mse'] = nonzero_reward_mse.item()
        Reward_loss['nonzero_reward_logprob'] = nonzero_reward_logprob.item()
        Reward_loss['nonzero_reward_num'] = nonzero_reward_num
        Reward_loss['zero_reward_mse'] = zero_reward_mse.item()
        Reward_loss['zero_reward_logprob'] = zero_reward_logprob.item()
        Reward_loss['zero_reward_num'] = zero_reward_num

        return Reward_loss

