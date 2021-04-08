import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from torch.distributions import Normal, kl_divergence

from mem_transformer import MemTransformer

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False,
                 mem_type='lstm', n_layer=5, n_head=8, dropout=0.0, mem_len=20,
                 beta_rep_kl=1.0, n_imagine=5, loss_type='agent-rep-img',
                 combine_loss=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.mem_type = mem_type
        self.beta_rep_kl = beta_rep_kl
        self.n_imagine = n_imagine
        self.loss_type = loss_type.split('-')
        self.combine_loss = combine_loss

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

        # Define memory
        if self.use_memory:
            # state + action
            memory_inp_size = self.semi_memory_size + 1
            if mem_type=='lstm':
                self.memory_module = nn.LSTMCell(memory_inp_size, self.semi_memory_size)
            elif 'trxl' in mem_type:
                if mem_type=='trxl':
                    self.memory_module = MemTransformer(
                            memory_inp_size, n_layer=n_layer, n_head=n_head,
                            d_model=self.semi_memory_size,
                            d_head=self.semi_memory_size//n_head,
                            d_inner=self.semi_memory_size,
                            dropout=dropout, dropatt=dropout, pre_lnorm=False,
                            tgt_len=1, ext_len=0, mem_len=mem_len, attn_type=0)
                elif mem_type=='trxli':
                    self.memory_module = MemTransformer(
                            memory_inp_size, n_layer=n_layer, n_head=n_head,
                            d_model=self.semi_memory_size,
                            d_head=self.semi_memory_size//n_head,
                            d_inner=self.semi_memory_size,
                            dropout=dropout, dropatt=dropout, pre_lnorm=True,
                            tgt_len=1, ext_len=0, mem_len=mem_len, attn_type=0)
                elif 'gtrxl' in mem_type:
                    gate = mem_type.split('-')[1]
                    self.memory_module = MemTransformer(
                            memory_inp_size, n_layer=n_layer, n_head=n_head,
                            d_model=self.semi_memory_size,
                            d_head=self.semi_memory_size//n_head,
                            d_inner=self.semi_memory_size,
                            dropout=dropout, dropatt=dropout, pre_lnorm=True,
                            tgt_len=1, ext_len=0, mem_len=mem_len, attn_type=0,
                            gate=gate)
                else:
                    raise ValueError("The TrXL must be one of trxl, trxli and gtrxls")
            else:
                raise ValueError("The TrXL must be lstm or trxls")

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

        # Define text embedding
        if self.use_text:
            raise ValueError("This version doesn't support the language embedded observation now.")
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Define image decoder
        self.image_deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, (3,3)),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, (3,3)),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 20, (3,3)),
                #nn.ReLU(),
                #nn.Conv2d(8, 32, (1,1)),
                #nn.ReLU(),
                #nn.Conv2d(32, 20, (1,1)), # color_idx: 6, obj_idx: 11, state_idx: 3
        )

        # Define reward decoder
        self.reward_decoder = nn.Sequential(
                nn.Linear(64+1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    #def img_decode(self, obs):
    def forward(self, obs, memory, prev_action, prev_state,
            actions=None, rewards=None, get_rep_loss=False, get_prior=False):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        # transition
        if self.use_memory:
            mem_inp = torch.cat([prev_action.unsqueeze(-1), prev_state], dim=-1)
            if self.mem_type=='lstm':
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_module(mem_inp, hidden)
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:  # transformers
                embedding, memory = self.memory_module(mem_inp.unsqueeze(0), *memory)
                embedding = embedding[0]
            # p(s_t | s_{t-1}, a_{t-1})
            prior_mean, prior_std = self.prior_net(embedding).split(
                    (self.semi_memory_size, self.semi_memory_size), dim=-1)
            prior_std = F.softplus(prior_std) + 0.1
            prior_dist = Normal(prior_mean, prior_std)
            prior = prior_dist.rsample()

            if get_prior:
                if not self.combine_loss:
                    prior = prior.detach()
                x = self.actor(prior)
                dist = Categorical(logits=F.log_softmax(x, dim=1))
                x = self.critic(prior)
                value = x.squeeze(1)
                return dist, value, memory, prior, None

        else:
            raise ValueError("Dreamer is memory-based model.")
            #embedding = x

        # q(s_t | s_{t-1}, a_{t-1}, o_t)
        post_mean, post_std = self.post_net(
                torch.cat([embedding, x], dim=-1)).split(
                (self.semi_memory_size, self.semi_memory_size), dim=-1)
        post_std = F.softplus(post_std) + 0.1
        post_dist = Normal(post_mean, post_std)
        post = post_dist.rsample()

        # s_t
        embedding = post

        if get_rep_loss:

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

            ## reward loss
            pred_reward = self.reward_decoder(
                    torch.cat([embedding, actions.unsqueeze(-1)], dim=-1))

            # nonzero reward loss
            nonzero_rewards = rewards[rewards!=0]
            nonzero_pred_reward = pred_reward[rewards!=0, :]
            nonzero_reward_num = len(nonzero_rewards)
            if nonzero_reward_num == 0:
                nonzero_reward_loss = torch.Tensor([0])
            else:
                nonzero_reward_loss = nn.MSELoss()(nonzero_pred_reward,
                    nonzero_rewards.unsqueeze(-1))

            # zero reward loss
            zero_rewards = rewards[rewards==0]
            zero_pred_reward = pred_reward[rewards==0, :]
            zero_reward_num = len(zero_rewards)
            if zero_reward_num == 0:
                zero_reward_loss = torch.Tensor([0])
            else:
                zero_reward_loss = nn.MSELoss()(zero_pred_reward,
                    zero_rewards.unsqueeze(-1))

            reward_loss = nn.MSELoss()(pred_reward, rewards.unsqueeze(-1))

            # KL
            kl_loss = kl_divergence(post_dist, prior_dist).mean()

            # representation loss
            rep_loss = {}
            rep_loss['rep_loss'] = recon_loss + reward_loss + self.beta_rep_kl*kl_loss
            rep_loss['recon_acc'] = recon_acc
            rep_loss['recon_loss'] = recon_loss
            rep_loss['recon_col_loss'] = col_loss
            rep_loss['recon_obj_loss'] = obj_loss
            rep_loss['recon_state_loss'] = state_loss
            rep_loss['reward_loss'] = reward_loss
            rep_loss['nonzero_reward_loss'] = nonzero_reward_loss.item()
            rep_loss['nonzero_reward_num'] = nonzero_reward_num
            rep_loss['zero_reward_loss'] = zero_reward_loss.item()
            rep_loss['zero_reward_num'] = zero_reward_num
            rep_loss['kl_loss'] = kl_loss

        else:
            rep_loss = None

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        if ('rep' in self.loss_type) and (not self.combine_loss):
            embedding = embedding.detach()

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory, embedding, rep_loss

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
