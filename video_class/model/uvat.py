import copy
import json
import logging
from io import open

import torch
from torch import nn
import sys
sys.path.append('../model/')

#from apex.normalization.fused_layer_norm import FusedLayerNorm
from torch.nn import LayerNorm as FusedLayerNorm
#from model.audio_embedding import UniterAudioEmbeddings
from model.audio_embedding_mfcc import UniterAudioEmbeddings
from model.text_embedding import UniterTextEmbeddings
#from model.video_embedding import UniterVideoEmbeddings
from model.video_embedding_max_patch import UniterVideoEmbeddings
#from model.video_embedding_image import UniterVideoEmbeddings
from model.bert import UniterPreTrainedModel, BertPooler, UniterEncoder


class UniterModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.text_embedding = UniterTextEmbeddings(config)
        self.video_embedding = UniterVideoEmbeddings(config)
        self.audio_embedding = UniterAudioEmbeddings(config)
        self.segment_embedding = nn.Embedding(3, config.hidden_size, padding_idx=0)
        self.encoder = UniterEncoder(config)
        
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)
        self.agg_token_embd = torch.nn.init.normal_(torch.nn.Parameter(torch.FloatTensor(768), requires_grad=True))
        self.fc = torch.nn.Linear(config.hidden_size, config.num_class)
        
    def _append_special_tokens(self, inputs):
        batch_size = inputs.shape[0]
        special_embds = self.agg_token_embd[None, None, :]
        # (batch_size, 1, d_model)
        special_embds = special_embds.repeat(batch_size, 1, 1)
        return torch.cat([special_embds, inputs], dim=1)


    def forward(self, img_feat, video_position_id, video_attn_mask, 
                audio_feat, audio_position_id, audio_attn_mask, 
                title_id, title_position_id, title_attn_mask,
                #query_id, query_position_id, query_attn_mask,
                gather_index=None, img_masks=None,
                output_all_encoded_layers=False,
                txt_type_ids=None, img_type_ids=None):
        # compute self-attention mask
        if len(img_feat) != 0:
            video_embedding, video_attn_mask = self.video_embedding(img_feat, video_position_id, video_attn_mask)
        else:
            video_embedding = torch.zeros(0, dtype=img_feat.dtype, device=img_feat.device)
            video_attn_mask = torch.zeros(0, dtype=video_attn_mask.dtype, device=img_feat.device)
        if len(audio_feat) != 0 :
            audio_embedding, audio_attn_mask = self.audio_embedding(audio_feat, audio_position_id, audio_attn_mask)
        else:
            audio_embedding = torch.zeros(0, dtype=torch.float, device=audio_feat.device)
            audio_attn_mask = torch.zeros(0, dtype=torch.long, device=audio_feat.device)
        if len(title_id) != 0:             
            title_embedding, title_attn_mask = self.text_embedding(title_id, title_position_id, title_attn_mask)
        else:
            title_embedding = torch.zeros(0, dtype=torch.float, device=audio_feat.device)
            title_attn_mask = torch.zeros(0, dtype=torch.long, device=audio_feat.device)

        #if len(query_id) != 0:             
        #    query_embedding, query_attn_mask = self.text_embedding(query_id, query_position_id, query_attn_mask)
        #else:
        #    query_embedding = torch.zeros(0, dtype=torch.float, device=audio_feat.device)
        #    query_attn_mask = torch.zeros(0, dtype=torch.long, device=audio_feat.device) 
        batchsize = video_attn_mask.shape[0] 
        #query_segment_label_len = query_attn_mask.shape[-1]
        #data_segment_label_len = title_attn_mask.shape[-1] + video_attn_mask.shape[-1] + audio_attn_mask.shape[-1]
        #segment_label = torch.tensor([1 for _ in range(query_segment_label_len)] + [2 for _ in range(data_segment_label_len)],\
        #                    dtype=torch.long, device=audio_feat.device).unsqueeze(0)
        #segment_embedding = self.segment_embedding(segment_label)     

        #all_embedding = torch.cat([query_embedding, video_embedding, audio_embedding, title_embedding], dim=1)
        all_embedding = torch.cat([video_embedding, audio_embedding, title_embedding], dim=1)
        all_embedding = all_embedding #+ segment_embedding
        all_embedding = self._append_special_tokens(all_embedding)
        cls_mask = torch.ones(1, dtype=video_attn_mask.dtype).to(title_id.device)
        cls_mask = cls_mask.repeat(batchsize, 1)
        attention_mask = torch.cat([cls_mask, video_attn_mask, audio_attn_mask, title_attn_mask], dim=1)
        #print(all_embedding.shape)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(torch.float)
        #    dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # # embedding layer
        # if input_ids is None:
        #     # image only
        #     embedding_output = self._compute_img_embeddings(
        #         img_feat, img_pos_feat, img_masks, img_type_ids)
        # elif img_feat is None:
        #     # text only
        #     embedding_output = self._compute_txt_embeddings(
        #         input_ids, position_ids, txt_type_ids)
        # else:
        #     embedding_output = self._compute_img_txt_embeddings(
        #         input_ids, position_ids,
        #         img_feat, img_pos_feat,
        #         gather_index, img_masks, txt_type_ids, img_type_ids)
        #print(all_embedding.shape,extended_attention_mask.shape)
        encoded_layers = self.encoder(
            all_embedding, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return self.fc(encoded_layers[:, 0, :])

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    config = json.load(open('../conf/uvat.json','r'))
    import utils
    config = utils.Config(config)
    mode = UniterModel(config)
    #torch.save(mode,'./model.pth')
    #audio= torch.rand(2,1,1600*30)
    #video=torch.rand(2,3,32,224,224)
    #input_ids = torch.arange(0, 10).view(2,5).to(torch.long)
    #position_ids = torch.arange(0, 10).view(2,5).to(torch.long)
    #print(mode(input_ids,position_ids,video,audio,1).shape)













    
