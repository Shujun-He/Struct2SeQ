import torch
import torch.nn as nn
import numpy as np
import math 
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution layer - only operates on past elements in the sequence
    Input: (batch_size, sequence_length, channels)
    Output: (batch_size, sequence_length, out_channels)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding_size = (kernel_size - 1) * dilation
        
        # Create regular 1D convolution
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,  # We'll handle padding manually
        )

    def forward(self, x):
        # Input is BxLxC, conv expects BxCxL
        x = x.transpose(1, 2)
        
        # Add padding to the left/past
        padding = torch.zeros(
            x.shape[0],  # batch
            x.shape[1],  # channels
            self.padding_size,  # padding size
            device=x.device,
            dtype=x.dtype
        )
        x_padded = torch.cat([padding, x], dim=-1)
        
        # Apply convolution
        out = self.conv(x_padded)
        
        # Return to BxLxC format
        return out.transpose(1, 2)


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class GraphConv(nn.Module):
    def __init__(self, input_dim, dropout):
        super(GraphConv, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*4, input_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x, adj_matrix):
        res = x
        adj_matrix = adj_matrix/adj_matrix.sum(-1,keepdim=True)
        x = torch.matmul(adj_matrix, x)
        x = x + res
        res = x
        x = self.layer_norm1(x)
        x = self.ffn(x)
        x = x + res
        x = self.layer_norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, db_vocab_size, embed_size, num_layers, nhead, dim_feedforward, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(db_vocab_size, embed_size)

        # self.positional_encoding = PositionalEncoding(
        #     embed_size, dropout=dropout)
        self.positional_encoding = nn.LSTM(embed_size,embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, 
                                                        dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)
        self.conv = nn.Sequential(nn.Conv1d(embed_size,embed_size,5,padding=2),
                                  nn.Dropout(dropout))

        self.gconv=GraphConv(embed_size,dropout)

    def forward(self, src, ct_matrix, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src)
        
        res = src
        src = self.dropout(self.positional_encoding(src)[0])
        src = self.conv(src.permute(0,2,1)).permute(0,2,1)
        src = self.norm(src+res)
        src = self.gconv(src,ct_matrix)
        # print(src.shape)
        # exit()
        
        return self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

class Decoder(nn.Module):
    def __init__(self, rna_vocab_size, embed_size, num_layers, nhead, dim_feedforward, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(rna_vocab_size, embed_size)
        self.paired_embedding = nn.Embedding(2, embed_size)
        # self.positional_encoding = PositionalEncoding(
        #     embed_size, dropout=dropout)
        self.positional_encoding = nn.LSTM(embed_size,embed_size)
        self.decoder_layers = nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, 
                                                        dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, rna_vocab_size-1)

    def forward(self, tgt, paired_encoding, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.embedding(tgt)
        
        #embed paired or not
        L=tgt.shape[1]-1
        tgt[:,1:]=tgt[:,1:]+self.paired_embedding(paired_encoding)[:,:L]

        tgt = self.positional_encoding(tgt)[0]

        output = self.transformer_decoder(tgt, memory, memory_mask=memory_mask, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return self.fc_out(output)

class OptimizedTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, nhead, dropout):
        super().__init__()
        embed_size = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            OptimizedDecoderLayer(d_model, nhead, dropout = dropout)
            for _ in range(num_layers)
        ])
        #self.norm = nn.LayerNorm(d_model)
        self.positional_encoding = nn.LSTM(embed_size,embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size-1)
        self.paired_embedding = nn.Embedding(2, embed_size)

        self.conv = nn.Sequential(CausalConv1d(embed_size,embed_size,5),
                                  nn.Dropout(dropout))

        self.conv_norm = nn.LayerNorm(embed_size)

    def forward(self, tgt, paired_encoding, memory, tgt_mask=None, past_key_values=None, use_cache=False):

        tgt = self.embedding(tgt)
        
        #print(tgt.shape)
        #embed paired or not
        L=tgt.shape[1]
        tgt=tgt+self.paired_embedding(paired_encoding)[:,:L]

        res = tgt
        tgt = self.positional_encoding(tgt)[0]
        tgt = self.conv(tgt)
        tgt = self.conv_norm(tgt+res)

        output = tgt
        new_key_values = [] if use_cache else None
        
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            #if use_cache:
            output, layer_key_values = layer(output, memory, tgt_mask, layer_past, use_cache)
            if use_cache:
                new_key_values.append(layer_key_values)
        
        #output = self.norm(output)
        output = self.fc_out(output)
        if use_cache:
            return output, new_key_values
        else:
            return output

class OptimizedDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, tgt, memory, tgt_mask=None, past_key_value=None, use_cache=False):
        # Self-attention
        # print(tgt.shape)
        # exit()
        if past_key_value is not None:
            tgt_res = tgt
            tgt_q = tgt[:,-1,None]
            tgt_k, tgt_v = past_key_value
            tgt_k = torch.cat([tgt_k, tgt_q], dim=1)
            tgt_v = torch.cat([tgt_v, tgt_q], dim=1)
            tgt_mask = None #no masking when doing kv cached decoding
            # print(tgt_q.shape)
            # print(tgt_k.shape)
            # print(tgt_v.shape)
            # exit()
        else:
            tgt_q, tgt_k, tgt_v = tgt, tgt, tgt

        self_attn_output, _ = self.self_attn(tgt_q, tgt_k, tgt_v, attn_mask=tgt_mask)

        tgt_q = self.norm1(tgt_q + self_attn_output)
        

        mem_k, mem_v = memory, memory
        
        cross_attn_output, _ = self.multihead_attn(tgt_q, mem_k, mem_v)
        tgt_q = self.norm2(tgt_q + cross_attn_output)
        
        # Feed-forward network
        ff_output = self.ffn(tgt_q)
        tgt_q = self.norm3(tgt_q + ff_output)

        if use_cache:
            tgt_q = torch.cat([tgt[:,:-1],tgt_q],1)
            # print(tgt.shape)
            # print(tgt_q.shape)
        else:
            pass
        
        #exit()
        if use_cache:
            return tgt_q, (tgt_k, tgt_v)
        else:
            return tgt_q, None


class DotBracketRNATransformer(nn.Module):
    def __init__(self, db_vocab_size, rna_vocab_size, embed_size, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(DotBracketRNATransformer, self).__init__()
        self.encoder = Encoder(db_vocab_size, embed_size, num_encoder_layers, nhead, dim_feedforward, dropout)
        self.decoder = OptimizedTransformerDecoder(rna_vocab_size, embed_size, num_decoder_layers, nhead, dropout)

    def forward(self, src, ct_matrix, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src, ct_matrix, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
        paired_encoding = (src==0).long()
        #output = self.decoder(tgt, paired_encoding, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_key_padding_mask, tgt_key_padding_mask=tgt_padding_mask)
        output = self.decoder(tgt, paired_encoding, memory, tgt_mask=tgt_mask)
        return output + memory.mean()*0

def generate_square_subsequent_mask(src):
    sz=src.shape[1]
    mask = (torch.triu(torch.ones((sz, sz), device=src.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    PAD_IDX=4
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=src.device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

