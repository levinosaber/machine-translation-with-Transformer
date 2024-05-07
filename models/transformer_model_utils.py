import torch.nn as nn
import torch
import math
import copy

from models.transformer_model_core import EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, \
MultiHeadedAttention, PositionwiseFeedForward, Embeddings, PositionalEncoding, Generator

def make_model(
    src_vocab_len, tgt_vocab_len, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    ''' 
    Construct a model from hyperparameters
    params:
        src_vocab: int, the size of source vocabulary
        tgt_vocab: int, the size of target vocabulary
        N:         int, the number of encoder and decoder layers
        d_model:   int, the dimension of model(embedding dimension)
        d_ff:      int, the dimension of positionwise feed forward layer
        h:         int, the number of heads in multi-head attention
        dropout:   float, the dropout rate
    '''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab_len), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab_len), c(position)),
        Generator(d_model, tgt_vocab_len),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model