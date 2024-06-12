import torch
from torch import nn
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

class AttentionEmbedding(nn.Module):
    def __init__(self, context_window, window, 
                 c_in=1, stride=8, d_attn=32, conv_stride=8,
                 n_layer=3, n_head=4, n_embd=16, alpha=0.9, 
                 initializer_range=0.2, embd_type='attention'):
        super().__init__()

        n_embd = n_head * 8

        self.n_global = ((context_window + stride) - conv_stride) // conv_stride + 1

        # Adjust the down_conv layer to handle multiple input channels
        self.down_conv = nn.Conv1d(c_in, c_in, conv_stride, stride=conv_stride)

        self.n_head = n_head
        self.context_window = context_window
        self.window = window
        self.stride = stride
        self.c_in = c_in
        self.d_attn = d_attn
        self.d_out = (window + self.n_global) * n_head * n_layer
        self.embd_type = embd_type
        self.n_segment = int((context_window - window) / stride + 1) + 1
        print("self.embd_type = {}".format(self.embd_type))
        if embd_type == 'attention':
            config = GPT2Config()
            config.n_layer = n_layer
            config.n_head = n_head
            config.n_embd = n_embd
            config.initializer_range = initializer_range
            config.cat_length = self.n_global
            config.ema_length = self.window
            config.alpha = alpha
            print("alpha = {}".format(alpha))
            self.attn_encoder = GPT2Model(config)
        elif embd_type == 'rbf':
            self.ln_A = nn.LayerNorm(n_embd)
            self.W_Q = nn.Linear(n_embd, n_embd)
            self.W_K = nn.Linear(n_embd, n_embd)
            self.sigma = torch.nn.Parameter(torch.rand(n_head) * 0.1)
        elif embd_type == 'poly':
            self.ln_A = nn.LayerNorm(n_embd)
            self.W_Q = nn.Linear(n_embd, n_embd)
            self.W_K = nn.Linear(n_embd, n_embd)
        
        self.W_A = nn.Linear(c_in * (window + self.n_global), n_embd)
        self.final_linear = nn.Linear(53 * 6, 64 * 8 * c_in)  # Adjust the final output dimension
        
    def fetch_attn_embd(self, x):
        attn = self.attn_encoder(inputs_embeds=x, output_attentions=True).attentions
        attn = torch.cat(attn, dim=1)
        attn = attn[:, :, -1, :]  # b h l

        return attn

    def fetch_rbf_embd(self, x):
        query = self.W_Q(x)
        query = rearrange(query,  'b w (h d) -> b h w d', h=self.n_head)[:, :, [-1], :]
        key = self.W_K(x)
        key = rearrange(key,  'b w (h d) -> b h w d', h=self.n_head)
        
        attn = (query - key).pow(2).sum(dim=-1)
        attn = attn * (self.sigma.unsqueeze(dim=-1))
        attn = torch.exp(attn)  # b h l

        return attn

    def fetch_poly_embd(self, x):
        query = self.W_Q(x)
        query = rearrange(query,  'b w (h d) -> b h w d', h=self.n_head)[:, :, [-1], :]
        key = self.W_K(x)
        key = rearrange(key,  'b w (h d) -> b h w d', h=self.n_head)

        attn = torch.einsum('bhsd,bhwd->bhw', query, key)
        attn = attn ** 4  # b h l

        return attn

    def forward(self, x):
        B, L, D = x.shape
        # print(0, x.shape)

        down_sample = self.down_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # print(1, down_sample.shape)
        
        emb_x = []
        for i in range(self.n_segment):
            st = i * self.stride
            end = st + self.window
            if end > L:
                break
            cur = x[:, st:end, :]
            cur = torch.cat([down_sample[:, :self.n_global, :], cur], dim=1)
            emb_x.append(cur.unsqueeze(dim=1))
            
        # print(2, emb_x[0].shape)
        
        emb_x = torch.cat(emb_x, dim=1).contiguous()  # B, N, W, D
        # print(3, emb_x.shape)
        emb_x = rearrange(emb_x, 'b n w d -> b n (w d)')
        # print(3.5, emb_x.shape)
        emb_x = self.W_A(emb_x)
        # print(4, emb_x.shape)
        if self.embd_type == 'attention':
            emb_x = self.fetch_attn_embd(emb_x)
        elif self.embd_type == 'rbf':
            emb_x = self.ln_A(emb_x)
            emb_x = self.fetch_rbf_embd(emb_x)
        elif self.embd_type == 'poly':
            emb_x = self.ln_A(emb_x)
            emb_x = self.fetch_poly_embd(emb_x)
        # print(5, emb_x.shape)

        # Flatten the last dimensions and pass through final linear layer
        emb_x = rearrange(emb_x, 'b h l -> b (h l)')
        emb_x = self.final_linear(emb_x)

        # Reshape to the desired output dimension: 64 x 7 x 8 x 64
        emb_x = rearrange(emb_x, 'b (c n p) -> b c n p', c=self.c_in, n=8, p=64)
        # print(7, emb_x.shape)
        return emb_x