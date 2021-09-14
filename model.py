import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

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
                 img_encode=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.mem_type = mem_type
        self.img_encode = img_encode

        # Define image embedding
        if img_encode:
          depth = 32
          act = nn.ELU
          kernels = (4, 4, 4, 4)
          layers = []
          for i, kernel in enumerate(kernels):
            if i == 0:
              inp_dim = 3
            else:
              inp_dim = 2 ** (i-1) * depth
            out_dim = 2 ** i * depth
            layers.append(nn.Conv2d(inp_dim, out_dim, kernel, 2))
            layers.append(act())
          self.image_conv = nn.Sequential(*layers)
          self.image_embedding_size = out_dim * 2 * 2
        else:
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
            if mem_type=='lstm':
                self.memory_module = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
            elif 'trxl' in mem_type:
                if mem_type=='trxl':
                    self.memory_module = MemTransformer(
                            self.image_embedding_size, n_layer=n_layer, n_head=n_head,
                            d_model=self.semi_memory_size,
                            d_head=self.semi_memory_size//n_head,
                            d_inner=self.semi_memory_size,
                            dropout=dropout, dropatt=dropout, pre_lnorm=False,
                            tgt_len=1, ext_len=0, mem_len=mem_len, attn_type=0)
                elif mem_type=='trxli':
                    self.memory_module = MemTransformer(
                            self.image_embedding_size, n_layer=n_layer, n_head=n_head,
                            d_model=self.semi_memory_size,
                            d_head=self.semi_memory_size//n_head,
                            d_inner=self.semi_memory_size,
                            dropout=dropout, dropatt=dropout, pre_lnorm=True,
                            tgt_len=1, ext_len=0, mem_len=mem_len, attn_type=0)
                elif 'gtrxl' in mem_type:
                    gate = mem_type.split('-')[1]
                    self.memory_module = MemTransformer(
                            self.image_embedding_size, n_layer=n_layer, n_head=n_head,
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

        # Define text embedding
        if self.use_text:
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

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            if self.mem_type=='lstm':
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_module(x, hidden)
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:  # transformers
                embedding, memory = self.memory_module(x.unsqueeze(0), *memory)
                embedding = embedding[0]
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
