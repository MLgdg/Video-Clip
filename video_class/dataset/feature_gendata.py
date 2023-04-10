"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from torch.utils.data import Dataset
import random
from scipy.io import wavfile
from dataset.tokenization import BertTokenizer
import json
from torchvision import transforms
import librosa
import os
import soundfile
import numpy as np
#from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index
import sys
from PIL import Image
sys.path.append('../')
from utils import MyThread, _build_vid_pos_ids
config = json.load(open('./conf/uvat.json','r'))
import utils
config = utils.Config(config)
root_path = config.root_path
def _get_vqa_target(example, num_answers):
    target = torch.zeros(num_answers)
    labels = example['target']['labels']
    scores = example['target']['scores']
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
    return target


class UVATDataset(Dataset):
    def __init__(self, data_file, config, sets='TRAIN'):
        super().__init__()
        self.data_file = data_file
        self.data = json.load(open(self.data_file, 'r'))
        self.data_names = list(self.data.keys())
        self.num_data = len(self.data)
        self.max_lenth = config.max_image
        self.sets = sets
        self.label2id = json.load(open(config.label2id, 'r'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.tokener = BertTokenizer.from_pretrained('./dataset/')
        #print(self.label2id)
        #print(len(self.data_names))
    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, i):
        file_name = self.data_names[i]
        file = self.data.get(file_name)
        #print(file)
        image, image_number = self.get_img_input(file)
        audio, audio_len = self.get_audio_input(file)
        text, text_len = self.get_text_input(file)
        
        if random.random() > 0.5:
            quert, query_len = self.get_quert_input(file)
            quert_label = 1
        else: 
            query_file_name = self.data_names[random.randrange(len(self.data_names))]
            query_file = self.data.get(query_file_name)
            quert, query_len = self.get_quert_input(query_file)
            quert_label = 0


        #print(self.label2id, file['label'])
        target = torch.tensor(int(self.label2id.get(file['label'])))

        image_patch_number = (image.shape[-1] // config.spatial_patch_size) ** 2 * (image_number // config.temporal_patch_size)
        audio_patch_number = audio_len // config.audio_temporal_patch_size
        image_attn_masks = torch.ones(image_patch_number, dtype=torch.long)
        audio_attn_masks = torch.ones(audio_patch_number, dtype=torch.long)
        text_attn_masks = torch.ones(text_len, dtype=torch.long)


        #img_feat, img_pos_feat, num_bb = self._get_img_feat(
        #    example['img_fname'])

        # text input
        #input_ids = example['input_ids']
        #input_ids = self.txt_db.combine_inputs(input_ids)

        #target = _get_vqa_target(example, self.num_answers)

        #attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return image, image_number, image_attn_masks, audio, audio_len, audio_attn_masks, text, text_len, text_attn_masks, target

    def process_single(self, img_path, img_names, i):
        img_name = img_names[i]
        image = Image.open(os.path.join(img_path, img_name)).convert('RGB')
        image = self.transform(image)
        #image = torch.Tensor(image)
        return image
    def get_img_input(self, info):

        images = list()
        if 'video_image_path' not in info:
            return torch.zeros(0, dtype=torch.float), 0
        #num_bb = min(info['num_frame'], self.max_lenth)
        img_path = os.path.join(root_path, info['video_image_path'])
        img_names = sorted(os.listdir(img_path))    # 假数据
        if len(img_names) >= config.temporal_patch_size:
            num_bb = (min(len(img_names), self.max_lenth) // config.temporal_patch_size) * config.temporal_patch_size
        #img_pos_feat = np.zeros([num_bb, self.max_lenth])
            threads = []
            for i in range(num_bb):
                #img_pos_feat[i, i] = 1  # 这里可能需要加个判断，是否是真的图像
                #img_pos_feat[i, i] = i * 1.0 / num_bb   # 这里是新改的，原来没有归一化
                thread = MyThread(self.process_single, args=(img_path, img_names, i))
                thread.start()
                threads.append(thread)
            for i in range(num_bb):
                thread = threads[i]
                image = thread.get_result()
                images.append(image)
            img_feat = torch.stack(images)
            #img_pos_feat = torch.Tensor(img_pos_feat)
            return img_feat, num_bb
        else:
            threads = []
            for i in range(len(img_names)):
                #img_pos_feat[i, i] = 1  # 这里可能需要加个判断，是否是真的图像
                #img_pos_feat[i, i] = i * 1.0 / num_bb   # 这里是新改的，原来没有归一化
                thread = MyThread(self.process_single, args=(img_path, img_names, i))
                thread.start()
                threads.append(thread)
            for i in range(len(img_names)):
                thread = threads[i]
                image = thread.get_result()
                images.append(image)
            for i in range(config.temporal_patch_size-len(img_names)):
                images.append(torch.zeros([3,224,224]))
            img_feat = torch.stack(images)
            #img_pos_feat = torch.Tensor(img_pos_feat)
            return img_feat, config.temporal_patch_size
            
    def get_audio_input(self, info):
        if 'video_audio_path' not in info:
            return torch.zeros(0, dtype=torch.float), 0
        #print(os.path.join(root_path, info['video_audio_path']))
        sr, wav = wavfile.read(os.path.join(root_path, info['video_audio_path']))
        wav = wav[: config.max_audio]
        wav = wav/(max(wav)+1e-7)
        audio_mfcc = torch.tensor(librosa.feature.mfcc(wav, sr, n_mfcc = 13))
        return audio_mfcc, audio_mfcc.shape[-1]

    def get_text_input(self, info):
        if 'title' not in info:
            return torch.zeros(0, dtype=torch.long), 0
        title = info.get('title')
        if not title:
            title = "视频无标题"
        text = self.tokener.tokenize(title)
        text_id = self.tokener.convert_tokens_to_ids(text)
        len_text = min(config.max_text_len, len(text_id))
        return torch.tensor(text_id[:len_text]), len_text
    def get_quert_input(self, info):
        if 'query' not in info:
            return torch.zeros(0, dtype=torch.long), 0, 0
        query = info.get('query') + '[SEP]'
        query = self.tokener.tokenize(query)
        query = self.tokener.convert_tokens_to_ids(query)
        query_len = min(config.max_text_len, len(query))
        return torch.tensor(query[:query_len]), query_len


def pad_tensors_img(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if len(tensors[0]) == 0:
        return torch.zeros(0, dtype=tensors[0].dtype)
    if lens is None:
        lens = [t.shape[0] for t in tensors]
    #print(len(tensors),tensors[0].shape)
    max_len = max(lens)
    bs = len(tensors)
    h, w = tensors[0].shape[2:] #tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, 3, h, w, dtype=dtype)

    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        # print('t, l: ', t.shape, l)
        output.data[i, :l, ..., ..., ...] = t.data
    return output
def pad_tensors_audio(tensors, lens=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.shape[0] for t in tensors]
    if len(tensors[0]) == 0:
        return torch.zeros(0, dtype=tensors[0].dtype) 
    max_len = max(lens)
    c, _ = tensors[0].shape
    bs = len(tensors)
    #h, w = tensors[0].shape[2:] #tensors[0].size(-1)
    dtype = tensors[0].dtype
    output = torch.zeros(bs, c, max_len, dtype=dtype)

    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        # print('t, l: ', t.shape, l)
        output.data[i, :, :l] = t.data
    return output#.unsqueeze(1)


spatial_patch_size = config.spatial_patch_size
temporal_patch_size = config.temporal_patch_size
audio_temporal_patch_size = config.audio_temporal_patch_size
def vqa_collate(inputs):
    #(input_ids, img_feats, img_pos_feats, attn_masks, targets
    # ) = map(list, unzip(inputs))
    image, image_number, image_attn_masks, audio, audio_len, audio_attn_masks,\
     text, text_len, text_attn_masks, targets = map(list, unzip(inputs))


    #txt_lens = [i.size(0) for i in input_ids]
    #print(text)
    if len(text[0]) != 0:
        input_ids = pad_sequence(text, batch_first=True, padding_value=0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)
        text_attn_masks = pad_sequence(text_attn_masks, batch_first=True, padding_value=0)
    else:
        input_ids = torch.zeros(0, dtype=torch.long)
        position_ids = torch.zeros(0, dtype=torch.long)
        text_attn_masks = torch.zeros(0, dtype=torch.long)
    #print(input_ids, text_attn_masks)
    targets = torch.stack(targets, dim=0)
    #print(targets)
    img_feat = pad_tensors_img(image, image_number)
    #img_feat = img_feat.permute(0, 2, 1, 3, 4).contiguous()
    if len(img_feat) !=0:
        bs, tmp_image_mask_shape, c, h, w = img_feat.shape
        img_feat = img_feat.permute(0, 2, 1, 3, 4).contiguous()
        image_patch_number = (tmp_image_mask_shape // temporal_patch_size) * ((w // spatial_patch_size) ** 2)
        video_attn_masks = torch.zeros([bs, image_patch_number], dtype=torch.long)
        for i, t in enumerate(image_attn_masks):
            video_attn_masks.data[i, :len(t)] = t.data
        video_position_ids = _build_vid_pos_ids(tmp_image_mask_shape // temporal_patch_size, w // spatial_patch_size, w // spatial_patch_size)
    else:
        video_position_ids = torch.zeros(0, dtype=torch.long)
        video_attn_masks = torch.ones(0, dtype=torch.long)

    audio_feat = pad_tensors_audio(audio, audio_len)
    if len(audio_feat) != 0:
        bs, _, tmp_audio_mask_shape = audio_feat.shape
        audio_patch_number = tmp_audio_mask_shape // audio_temporal_patch_size
        audio_attn_masks_ = torch.zeros([bs, audio_patch_number], dtype=torch.long)
        for i, t in enumerate(audio_attn_masks):
            audio_attn_masks_.data[i, :len(t)] = t.data
        audio_position_ids = torch.arange(0, audio_patch_number, dtype=torch.long).unsqueeze(0)
    else:
        audio_attn_masks_ = torch.zeros(0, dtype=torch.long)
        audio_position_ids = torch.ones(0, dtype=torch.long)

    #bs, max_tl = input_ids.size()
    #out_size = attn_masks.size(1)
    #gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'text_attn_masks': text_attn_masks, 
             'img_feat': img_feat,
             'video_position_ids': video_position_ids,
             'video_attn_masks': video_attn_masks,
             'audio_feat': audio_feat,
             'audio_position_ids': audio_position_ids,
             'audio_attn_masks': audio_attn_masks_,
             'targets': targets}
    return batch


# class VqaEvalDataset(VqaDataset):
#     def __getitem__(self, i):
#         qid = self.ids[i]
#         example = DetectFeatTxtTokDataset.__getitem__(self, i)
#         img_feat, img_pos_feat, num_bb = self._get_img_feat(
#             example['img_fname'])

#         # text input
#         input_ids = example['input_ids']
#         input_ids = self.txt_db.combine_inputs(input_ids)

#         if 'target' in example:
#             target = _get_vqa_target(example, self.num_answers)
#         else:
#             target = None

#         attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

#         return qid, input_ids, img_feat, img_pos_feat, attn_masks, target


# def vqa_eval_collate(inputs):
#     (qids, input_ids, img_feats, img_pos_feats, attn_masks, targets
#      ) = map(list, unzip(inputs))

#     txt_lens = [i.size(0) for i in input_ids]

#     input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
#     position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
#                                 ).unsqueeze(0)
#     attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
#     if targets[0] is None:
#         targets = None
#     else:
#         targets = torch.stack(targets, dim=0)

#     num_bbs = [f.size(0) for f in img_feats]
#     img_feat = pad_tensors(img_feats, num_bbs)
#     img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

#     bs, max_tl = input_ids.size()
#     out_size = attn_masks.size(1)
#     gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

#     batch = {'qids': qids,
#              'input_ids': input_ids,
#              'position_ids': position_ids,
#              'img_feat': img_feat,
#              'img_pos_feat': img_pos_feat,
#              'attn_masks': attn_masks,
#              'gather_index': gather_index,
#              'targets': targets}
#     return batch
