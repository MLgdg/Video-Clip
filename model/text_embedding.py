import copy
import json
import logging
from io import open

import torch
from torch import nn
#from apex.normalization.fused_layer_norm import FusedLayerNorm
from torch.nn import LayerNorm as FusedLayerNorm


class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)


        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, text_attn_masks, token_type_ids=None):
        #if token_type_ids is None:
        #    token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings + position_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, text_attn_masks
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    config = json.load(open('../conf/uvat.json','r'))
    import utils
    config = utils.Config(config)
    mode = UniterTextEmbeddings(config)
    a = torch.arange(0, 10).view(2,5).to(torch.long)
    b = torch.arange(0, 10).view(2,5).to(torch.long)
    print(mode(a,b).shape)