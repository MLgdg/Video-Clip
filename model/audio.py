'''
@Author: your name
@Date: 2022-01-11 17:29:05
@LastEditTime: 2022-01-13 14:42:19
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /gaoqingdong/图短小融合/model/UVAT/test2/UVAT/model/audio_embedding_mfcc.py
'''
import copy
import json
import logging
from io import open
import random
import torch
from torch import nn
#from apex.normalization.fused_layer_norm import FusedLayerNorm
from torch.nn import LayerNorm as FusedLayerNorm
import librosa

class UniterAudioEmbeddings(nn.Module):
    """
    考虑极限情况下的特征维度
    embedding patch操作音频信号最大支持5分钟音频
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        #self.max_num_patches = int(config.patch_sampling_rate * max_positional_buckets)
        self.wave_to_patch = torch.nn.Conv1d(in_channels=13, out_channels=config.hidden_size,
                                             kernel_size=config.audio_temporal_patch_size, stride = config.audio_temporal_patch_size)
        self.audio_embeddings = nn.Embedding(config.audio_max_temporal_buckets,
                                            config.hidden_size, padding_idx=0)
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def _flatten_inputs(self, inputs):
        input_shape = inputs.shape
        bs = input_shape[0]
        d_embd = input_shape[1]
        inputs = inputs.permute(0, 2, 1).contiguous()
        return inputs, input_shape

    def _random_patch_selection(self, inputs, audio_attn_masks):
        training = self.training
        if training:
            batch_size, seq_len, dim = inputs.shape
            temporal_idx = torch.LongTensor(sorted(random.sample(range(seq_len), int(seq_len * self.config.patch_sampling_rate)))).to(inputs.device)
            inputs = inputs.index_select(1, temporal_idx)
            audio_attn_masks = audio_attn_masks.index_select(1, temporal_idx)
            #inputs_shape = [batch_size, int(seq_len * self.config.patch_sampling_rate), dim]
            return inputs, audio_attn_masks
        return inputs, audio_attn_masks

    def forward(self, inputs, audio_position_ids, audio_attn_masks, token_type_ids=None):
        #print(inputs)
        #inputs = self.LayerNorm(inputs)
        embeddings = self.wave_to_patch(inputs)
        embeddings, input_shape = self._flatten_inputs(embeddings)
        #temporal_position_ids = torch.arange(input_shape[-1])
        #print(input_shape)
        position_embeddings = self.audio_embeddings(audio_position_ids)
        embeddings = embeddings + position_embeddings
        #position_embeddings = self.LayerNorm(position_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #embeddings, audio_attn_masks = self._random_patch_selection(embeddings, audio_attn_masks) #去掉DropToken
        return embeddings, audio_attn_masks

# if __name__ == '__main__':
#     #librosa.feature.mfcc()
#     # import sys
#     # sys.path.append('../')
#     # config = json.load(open('../conf/uvat.json','r'))
#     # import utils
#     # config = utils.Config(config)
#     # mode = UniterAudioEmbeddings(config)
#     # a = torch.rand(2, 1, 16000*30)
#     # print(mode(a).shape)