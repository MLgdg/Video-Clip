import copy
import json
import logging
from io import open
import random
import torch
from torch import nn
#from apex.normalization.fused_layer_norm import FusedLayerNorm
from torch.nn import LayerNorm as FusedLayerNorm

class UniterVideoEmbeddings(nn.Module):
    """
    最大支持32秒视频
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        patch_stack = (config.temporal_patch_size, config.spatial_patch_size, config.spatial_patch_size)
        self.voxel_to_patch = torch.nn.Conv3d(in_channels=3, out_channels=config.hidden_size,\
                                kernel_size=patch_stack, stride = patch_stack)
        max_positional_buckets = (config.max_temporal_buckets#9
                              * config.max_vertical_buckets#14
                              * config.max_horizontal_buckets)#14
        self.max_num_patches = int(config.patch_sampling_rate * max_positional_buckets)
        #self.shape_resize = torch.
        self.max_temporal_positions = config.max_temporal_buckets
        self.max_vertical_positions = config.max_vertical_buckets
        self.max_horizontal_positions = config.max_horizontal_buckets

        self.temporal_embeddings = nn.Embedding(self.max_temporal_positions,
                                            config.hidden_size, padding_idx=0)
        self.vertical_embeddings = nn.Embedding(self.max_vertical_positions,
                                                config.hidden_size)
        self.horizontal_embeddings = nn.Embedding(self.max_horizontal_positions,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def _flatten_inputs(self, inputs):
        input_shape = inputs.shape
        bs = input_shape[0]
        d_embd = input_shape[1]
        inputs = inputs.view(bs, d_embd, -1)
        inputs = inputs.permute(0, 2, 1).contiguous()
        #inputs = inputs.permute(0, 2, 3, 0) 
        return inputs, input_shape

    # def _build_vid_pos_ids(self, t, h, w):
    #     temporal_ids = torch.arange(t)[:, None, None]  # (t, 1, 1)
    #     vertical_ids = torch.arange(h)[None, :, None]  # (1, h, 1)
    #     horizontal_ids = torch.arange(w)[None, None, :]  # (1, 1, w)
    #     temporal_ids = temporal_ids.repeat(1, h, w)  # (t, h, w)
    #     vertical_ids = vertical_ids.repeat(t, 1, w)  # (t, h, w)
    #     horizontal_ids = horizontal_ids.repeat(t, h, 1)  # (t, h, w)
    #     pos_ids = torch.stack([temporal_ids, vertical_ids, horizontal_ids], dim=3)
    #     pos_ids = torch.reshape(pos_ids, [-1, 3])  # (t*h*w, 3)
    #     # if next(self.parameters()).is_cuda:
    #     #     return pos_ids.cuda()
    #     # else:
    #     return pos_ids

    def _random_patch_selection(self, inputs, video_attn_masks):
        training = self.training
        if training:
            batch_size, seq_len, dim = inputs.shape
            #temporal_idx = torch.arange(seq_len)
            #temporal_idx = tf.random.shuffle(temporal_idx)[None, :]

            temporal_idx = torch.LongTensor(sorted(random.sample(range(seq_len), int(seq_len * self.config.patch_sampling_rate)))).to(inputs.device)
            inputs = inputs.index_select(1,temporal_idx)
            #inputs_shape = [batch_size, int(seq_len * self.config.patch_sampling_rate), dim]
            video_attn_masks = video_attn_masks.index_select(1,temporal_idx)
            return inputs, video_attn_masks
            #temporal_idx = torch.randperm(seq_len)[None, :]
            #temporal_idx = temporal_idx.repeat(batch_size, 1)
            #batch_idx = torch.arange(batch_size)[:, None]
            #batch_idx = batch_idx.repeat(1, seq_len)#张量扩充
            #gather_idx = torch.stack([batch_idx, temporal_idx], axis=2)
            #inputs = tf.gather_nd(inputs, gather_idx)[:, :self.max_num_patches, :]
            #input_shape = [batch_size, self.max_num_patches, dim]
        return inputs, video_attn_masks
        
    def forward(self, inputs, video_position_ids, video_attn_masks, token_type_ids=None):
        embeddings = self.voxel_to_patch(inputs)
        embeddings, input_shape = self._flatten_inputs(embeddings)
        #random_mask = torch.ones((embeddings.shape[0:2]), dtype=torch.float32)
        #pos_ids = self._build_vid_pos_ids(input_shape[-3], input_shape[-2], input_shape[-1])
        temporal_position_ids = video_position_ids[None, :, 0]
        vertical_position_ids = video_position_ids[None, :, 1]
        horizontal_position_ids = video_position_ids[None, :, 2]
        temporal_embeddings = self.temporal_embeddings(temporal_position_ids)
        #print(temporal_embeddings.shape)
        vertical_embeddings = self.vertical_embeddings(vertical_position_ids)
        horizontal_embeddings = self.horizontal_embeddings(horizontal_position_ids)
        embeddings = embeddings + temporal_embeddings + vertical_embeddings + horizontal_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings, video_attn_masks = self._random_patch_selection(embeddings, video_attn_masks)
        return embeddings, video_attn_masks
if __name__ == '__main__':
    import sys
    sys.path.append('../')
    config = json.load(open('../conf/uvat.json','r'))
    import utils
    config = utils.Config(config)
    mode = UniterVideoEmbeddings(config)
    torch.save(mode,'./video.pth')
    a = torch.rand(2,3,32,224,224)
    print(mode(a).shape)






