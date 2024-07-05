# This file contains the code of our model.
# Instantiate class MeKoo to get our model.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.multihead_attn(None, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask)

        return output

def make_position_matrix(position, d_model):
    """
    :param position: A tensor containing the positions
    :param d_model: The feature size
    :return: A tensor of shape (1, position.size(0), d_model)
    """
    return torch.zeros(1, position.size(0), d_model).scatter(2, position.view(-1, 1), torch.ones(position.size(0), 1))

def add_positional_encoding(seq, model_dim, max_len=5000):
    """
    :param seq: A tensor of shape (batch_size, seq_len)
    :param model_dim: The feature size of the model
    :param max_len: The maximum possible length of a sequence
    :return: A tensor of shape (batch_size, seq_len, model_dim)
    """
    seq_len = seq.size(1)
    position = torch.arange(0, seq_len, dtype=torch.long, device=seq.device)
    position = position.unsqueeze(0)  # For each sequence element, its positional encoding will be of size (1, seq_len)
    return seq + make_position_matrix(position, model_dim)

class TranEncoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 d_model=512, 
                 nhead=8, 
                 num_layers=2, 
                 dim_feedforward=2048,
                 dropout=0.1):
        super(TranEncoder, self).__init__()
        self.d_model = d_model
        # self.in_linear = nn.Linear(in_dim, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, seq):
        # seq = self.in_linear(seq)
        seq = add_positional_encoding(seq, self.d_model)
        output = self.transformer_encoder(seq)
        return output

class EncoderMLP(nn.Module):
    def __init__(self, 
                 in_dim,
                 d_model=512,
                 num_layers=2):
        super(EncoderMLP, self).__init__()
        linears = [nn.Linear(in_dim+d_model, d_model)] + [nn.Linear(d_model, d_model) for _ in range(num_layers-1)]
        self.MLP = nn.Sequential(*linears)

    def forward(self, x, T):
        z = torch.cat([x, T], dim=-1)
        return self.MLP(z)

class MeKooMatching(nn.Module):
    def __init__(self, d_model, num_k):
        super(MeKooMatching, self).__init__()
        self.K_pool = nn.Parameter(torch.randn(num_k, d_model, d_model))

    def forward(self, seq):
        seq1 = seq[:, :-1, :]
        seq2 = seq[:, 1:, :]
        seq2_hats = torch.einsum("btd,nde->bnte", seq1, self.K_pool)
        exp_error = torch.exp(torch.abs(seq2_hats-seq2.unsqueeze(1))) # bntd 
        lambdas = torch.sum(exp_error, dim=(0,2,3))/torch.sum(exp_error) # [n]
        K = torch.einsum("kde,k->de", self.K_pool, lambdas)

def koopmanFore(seq, len, K):
    fore = []
    last = seq[:,-1:, :]
    for _ in range(len):
        last = last * K
        fore.append(last)
    return torch.cat(fore, dim=1)

class Level(nn.Module):
    def __init__(self, in_dim, out_len, d_model=512, num_k=10, num_mlp_layers=2):
        super(Level, self).__init__()
        self.out_len = out_len
        self.encoder_mlp = EncoderMLP(in_dim, d_model, num_mlp_layers)
        self.mekoo = MeKooMatching(d_model, num_k)

    def forward(self, seq, T):
        M = self.encoder_mlp(seq, T)
        K = self.mekoo(M)
        return M, koopmanFore(seq, self.out_len, K)
    
class MeKoo(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_len,
                 num_levels=5,
                 d_model=512,
                 num_k=10,
                 num_mlp_layers=2,
                 num_attention_layers=2):
        super(MeKoo, self).__init__()
        self.in_linear = nn.Linear(in_dim, d_model)
        self.tranEncoder = TranEncoder(d_model, d_model, num_layers=num_attention_layers)
        self.levels = nn.ModuleList([Level(d_model, out_len, d_model=d_model,num_k=num_k, num_mlp_layers=num_mlp_layers) for _ in range(5)])
        self.out_linear = nn.Linear(d_model, in_dim)

        def forward(self, seq):
            seq = self.in_linear(seq)
            T = self.tranEncoder(seq)
            result = None
            for level in self.levels:
                seq, fore = level(seq, T)
                if result is None:
                    result = fore
                else:
                    result = result + fore
            return self.out_linear(result)
