# -*- coding: utf-8 -*-
"""
#####################################################################
    > File Name: wflow_dataset.py
    > Author: 
    > Email: @baidu.com
    > Created Time: 2022/01/27 19:06:44
    > Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#####################################################################
"""
import time
from wflow_sdk.core.dataset import WFlowDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import random
import json
import torch
import sys
import librosa
sys.path.append('../')
from dataset.tokenization import BertTokenizer
from utils import _build_vid_pos_ids
config = json.load(open('./conf/uvat.json','r'))
import utils
config = utils.Config(config)
import numpy as np

with open('./data/quert_data_train.json')as ff:
    for ll in ff:
        query_nid = json.loads(ll.strip())
num=0
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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
max_num_image = config.max_image
def fn(d):
    print(d[0].keys())
    #print(d[0]["DATA_VIDEO_FRAMES"][0])
    #print(type(d[0]["DATA_VIDEO_FRAMES"]))
    #print(len(d[0]["DATA_VIDEO_FRAMES"]))
    #print(d[0]["DATA_VIDEO_FRAMES"])
    #print(d[0]["DATA_VIDEO_FRAMES"][0].shape)
    #print(len(d[0]["DATA_VIDEO_FRAMES"]))
    return d
with open('./data/quert_data_train.json')as ff:
    for ll in ff:
        query_nid = json.loads(ll.strip())
tokener = BertTokenizer.from_pretrained('./dataset')
#dict_keys(['DATA_NID', 'DATA_AUDIO', 'DATA_TAG', 'DATA_CATEGORY', 'DATA_SUB_CATEGORY', 'DATA_VIDEO_FRAMES'])
def get_query_label_mask(nid):
    
    if nid in query_nid:
        if isinstance(query_nid[nid], str):
            query = eval(query_nid[nid])
        else:
            quert = query_nid[nid]
        #print('nid query:',query)
        if random.random() >0.48:
            #print('random smple:',random.sample(query, 1))
            #print('type:',type(query_nid[nid]))
            query = random.sample(query, 1)[0]
            query_data_label = 1
            #print('非随机query:',query)
        else:
            random_key = random.sample(query_nid.keys(), 1)[0]
            query = random.sample(eval(query_nid[random_key]),1)[0]
            #query = random.sample(random.sample(query_nid.keys(), 1)[0], 1)[0]
            #print('query 随机1',query)
            query_data_label = 0
    else:
        random_key = random.sample(query_nid.keys(), 1)[0]
        query = random.sample(eval(query_nid[random_key]),1)[0]
        #print('query 随机2',query)
        query_data_label = 0
    query = tokener.tokenize(query)
    query = query + ['[SEP]']
    query = tokener.convert_tokens_to_ids(query)
    query_len = min(config.max_text_len, len(query))

    return torch.tensor(query[:query_len]), torch.tensor(query_data_label), torch.ones(query_len, dtype=torch.long)

def get_mask_title(title):
    #print(title)
    if len(title)<1:
        title = "视频无标题"
        text = tokener.tokenize(title)
        text_id = tokener.convert_tokens_to_ids(text)
        len_title = min(config.max_text_len, len(text_id))
        return torch.tensor(text_id[:len_title]), torch.ones(len_title, dtype=torch.long)
    
    text = tokener.tokenize(title)
    for i in range(len(text)):
        if random.random() < 0.5:
            text[i] = '[MASK]'
    text_id = tokener.convert_tokens_to_ids(text)     
    len_title = min(config.max_text_len, len(text_id))
    return torch.tensor(text_id[:len_title]), torch.ones(len_title, dtype=torch.long)
def get_mask_audio(wav):
    try:
        s = time.time()
        wav = wav[: config.max_audio]
        #wav = wav[: 13*100]
        s2 = time.time()
        wav = wav/(max(wav)+1e-7)
        s3 = time.time()
        #audio_mfcc = wav.reshape((13, -1))
        #audio_mfcc = np.ones((13, 100))
        #print(audio_mfcc.shape)
        audio_mfcc = librosa.feature.mfcc(wav, sr=16000, n_mfcc = 13)
        audio_mfcc = torch.tensor(audio_mfcc).to(torch.float32)
        #audio_mfcc = torch.tensor(audio_mfcc)
        #print("dtype", audio_mfcc.dtype)
        #audio_mfcc = torch.tensor(audio_mfcc, dtype=torch.float32)
        #print(audio_mfcc.shape)
        e = time.time()
        #print(e - s, "e - s audio", s2 - s, s3 - s2, e - s3, audio_mfcc.dtype, audio_mfcc.shape)
        return audio_mfcc, audio_mfcc.shape[-1]
    except Exception as e:
        print(e)
        #print(wav)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def get_mask_video(video):
    imgs = []
    for img in video:
        #print(type(img))
        #print(img.shape)
        imgs.append(transform(img))
    if len(imgs) >= config.temporal_patch_size:
        num = int(len(imgs) // config.temporal_patch_size * config.temporal_patch_size)
        imgs = imgs[:num]
    else:
        for i in range(config.temporal_patch_size-len(imgs)):
            imgs.append(torch.zeros([3,224,224]))
            num = config.temporal_patch_size
    imgs = torch.stack(imgs)
    return imgs, num

import time
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=8)
executor2 = ThreadPoolExecutor(max_workers=8)
executor3 = ThreadPoolExecutor(max_workers=8)
executor4 = ThreadPoolExecutor(max_workers=8)

import gc
def vqa_collate_no(inputs):
    #return inputs
    #del inputs
    gc.collect()
    return torch.ones((10,10))

def collate0(inputs):
    s = time.time()
    #print("start vqa_collate")
    #(input_ids, img_feats, img_pos_feats, attn_masks, targets
    # ) = map(list, unzip(inputs))
    data_audios = []
    data_tags = []
    data_nids = []
    data_video_frames = []

    input_len = len(inputs)
    audio_mfccs = []
    len_audios = []
    title_ids = []
    title_masks  = []
    query_ids = []
    query_data_labels = []
    query_masks = []
    tag1s = []
    tag2s = []
    videos = []
    len_videos = []

    for input in inputs:
        ss = time.time()
        nid = input.get('DATA_NID')
        data_audio = input.get('DATA_AUDIO')
        #audio_mfcc, len_audio = get_mask_audio(input.get('DATA_AUDIO'))
        #title_id, title_mask = get_mask_title(input.get('DATA_TAG'))
        data_tag = input.get('DATA_TAG')
        ss2 = time.time()
        #query_id, query_data_label, query_mask = get_query_label_mask(nid)
        ss3 = time.time()
        tag1 = input.get('DATA_CATEGORY')
        tag2 = input.get('DATA_SUB_CATEGORY')
        #video, len_video = get_mask_video(input.get('DATA_VIDEO_FRAMES', [])[:max_num_image])
        data_video_frame = input.get('DATA_VIDEO_FRAMES', [])[:max_num_image]
        ee = time.time()
        #print("get_mask_video", ee - ss)
        data_nids.append(nid)
        data_audios.append(data_audio)
        data_video_frames.append(data_video_frame)
        #audio_mfccs.append(audio_mfcc)
        #len_audios.append(len_audio)
        data_tags.append(data_tag)
        #title_ids.append(title_id)
        #title_masks.append(title_mask)
        #query_ids.append(query_id)
        #query_data_labels.append(query_data_label)
        #query_masks.append(query_mask)
        #videos.append(video)
        #len_videos.append(len_video)
        ee = time.time()
        #print("for time" , ee - ss, ss2 - ss, ss3-ss2)

    s2 = time.time()
    s22 = time.time()
    results = executor.map(get_mask_audio, data_audios)
    results2 = executor.map(get_mask_video, data_video_frames)
    results3 = executor.map(get_mask_title, data_tags)
    results4 = executor.map(get_query_label_mask, data_nids)
    s3 = time.time()
    for result in results:
        audio_mfcc, len_audio = result
        audio_mfccs.append(audio_mfcc)
        len_audios.append(len_audio)
    s4 = time.time()
    for result in results2:
        video, len_video = result
        videos.append(video)
        len_videos.append(len_video)
    s5 = time.time()
    for result in results3:
        title_id, title_mask = result
        title_ids.append(title_id)
        title_masks.append(title_mask)
    s6 = time.time()
    for result in results4:
        query_id, query_data_label, query_mask = result
        query_ids.append(query_id)
        query_data_labels.append(query_data_label)
        query_masks.append(query_mask)
    del inputs
    gc.collect()
    outputs = {
        "audio_mfccs": audio_mfccs,
        "len_audios": len_audios,
        "videos": videos,
        "len_videos": len_videos,
        "title_ids": title_ids,
        "title_masks": title_masks,
        "query_ids": query_ids,
        "query_data_labels": query_data_labels,
        "query_masks": query_masks,

    }
    return outputs


def vqa_collate(feats):
    
    audio_mfccs = []
    len_audios = []
    videos = [] 
    len_videos = []
    title_ids = []
    title_masks = []
    query_ids = []
    query_data_labels = []
    query_masks = []
    for feat in feats:
        audio_mfccs.extend(feat.get("audio_mfccs"))
        len_audios.extend(feat.get("len_audios"))
        videos.extend(feat.get("videos"))
        len_videos.extend(feat.get("len_videos"))
        title_ids.extend(feat.get("title_ids"))
        title_masks.extend(feat.get("title_masks"))
        query_ids.extend(feat.get("query_ids"))
        query_data_labels.extend(feat.get("query_data_labels"))
        query_masks.extend(feat.get("query_masks"))
    s7 = time.time()
    input_len = len(audio_mfccs)
    #txt_lens = [i.size(0) for i in input_ids]
    #print(text)
    if len(title_ids[0]) != 0:
        title_ids = pad_sequence(title_ids, batch_first=True, padding_value=0)
        title_position_ids = torch.arange(0, title_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0).repeat(input_len,1)
        title_attn_masks = pad_sequence(title_masks, batch_first=True, padding_value=0)
    else:
        title_ids = torch.zeros(0, dtype=torch.long)
        title_position_ids = torch.zeros(0, dtype=torch.long)
        title_attn_masks = torch.zeros(0, dtype=torch.long)
    
    if len(query_ids[0]) != 0:
        query_ids = pad_sequence(query_ids, batch_first=True, padding_value=0)
        query_position_ids = torch.arange(0, query_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0).repeat(input_len,1)
        query_attn_masks = pad_sequence(query_masks, batch_first=True, padding_value=0)
    else:
        query_ids = torch.zeros(0, dtype=torch.long)
        query_position_ids = torch.zeros(0, dtype=torch.long)
        query_attn_masks = torch.zeros(0, dtype=torch.long)
    #print(input_ids, text_attn_masks)

    query_data_labels = torch.stack(query_data_labels, dim=0)
    #print(targets)
    img_feat = pad_tensors_img(videos, len_videos)
    #img_feat = img_feat.permute(0, 2, 1, 3, 4).contiguous()
    if len(img_feat) !=0:
        bs, tmp_image_mask_shape, c, h, w = img_feat.shape
        img_feat = img_feat.permute(0, 2, 1, 3, 4).contiguous()
        image_patch_number = (tmp_image_mask_shape // temporal_patch_size) * ((w // spatial_patch_size) ** 2)
        video_attn_masks = torch.zeros([bs, image_patch_number], dtype=torch.long)
        for i, t in enumerate(len_videos):
            tmp_patch_num = (t // temporal_patch_size) * ((w // spatial_patch_size) ** 2)
            video_attn_masks.data[i, :tmp_patch_num] = 1
        video_position_ids = _build_vid_pos_ids(tmp_image_mask_shape // temporal_patch_size, w // spatial_patch_size, w // spatial_patch_size).repeat(input_len, 1, 1)
    else:
        video_position_ids = torch.zeros(0, dtype=torch.long)
        video_attn_masks = torch.ones(0, dtype=torch.long)

    audio_feat = pad_tensors_audio(audio_mfccs, len_audios)
    if len(audio_feat) != 0:
        bs, _, tmp_audio_mask_shape = audio_feat.shape
        audio_patch_number = tmp_audio_mask_shape // audio_temporal_patch_size
        audio_attn_masks_ = torch.zeros([bs, audio_patch_number], dtype=torch.long)
        for i, t in enumerate(len_audios):
            tmp_patch_num = t//audio_temporal_patch_size
            audio_attn_masks_.data[i, :audio_temporal_patch_size] = 1
        audio_position_ids = torch.arange(0, audio_patch_number, dtype=torch.long).unsqueeze(0).repeat(input_len,1)
    else:
        audio_attn_masks_ = torch.zeros(0, dtype=torch.long)
        audio_position_ids = torch.ones(0, dtype=torch.long)

    #bs, max_tl = input_ids.size()
    #out_size = attn_masks.size(1)
    #gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    e = time.time()
    #print("end vqa_collate")
    gc.collect()

    batch = {'title_id': title_ids,
             'title_position_id': title_position_ids,
             'title_attn_mask': title_attn_masks, 
             'query_id':query_ids,
             'query_position_id':query_position_ids,
             'query_attn_mask':query_attn_masks,
             'img_feat': img_feat,
             'video_position_id': video_position_ids,
             'video_attn_mask': video_attn_masks,
             'audio_feat': audio_feat,
             'audio_position_id': audio_position_ids,
             'audio_attn_mask': audio_attn_masks_,
             'target': query_data_labels}
    #print("clo time", e - s, s2 - s, s22 - s2, s3 - s22, s4 - s3, s5 - s4, s6 - s5, s7 - s6, e - s7)
    #print(time.time())
    return batch
if __name__ == "__main__":
    train_dataset = WFlowDataset(name="duanxiao_video_train", version=1)
    train_dataloader = DataLoader(train_dataset, batch_size=5,collate_fn=vqa_collate, pin_memory=True)
    #print("nids | video_frames_num | video_frames[0].shape | label")
    for ret in train_dataloader:
        print(ret.keys())
        print("title_id:",ret['title_id'])
        print("title_position_id:",ret['title_position_id'])
        print("title_attn_mask:",ret['title_attn_mask'])
        print("query_id:",ret['query_id'])
        print("query_position_id:",ret['query_position_id'])
        print('query_attn_mask:',ret['query_attn_mask'])
        print("img_feat shape:",ret['img_feat'].shape)
        print("video_position_id shape:",ret['video_position_id'].shape)
        print("video_attn_mask shape:",ret['video_attn_mask'].shape)
        print("audio_feat shape:",ret['audio_feat'].shape)
        print("audio_position_id shape:",ret['audio_position_id'].shape)
        print("audio_attn_mask shape:",ret['audio_attn_mask'].shape)
        print("target:",ret['target'])
        break
    #    for nid in ret["DATA_NID"]:
    #        print(nid)
    #        if nid not in query_nid:
    #            print(nid)
    #        if num%10000==0:
    #            print(num)
        #video_frames_num = ret["DATA_VIDEO_FRAMES_NUM"]
        #video_frames = ret["DATA_VIDEO_FRAMES"]
        #label = ret["DATA_LABEL"]
        #print(nids, "|", video_frames_num, "|", video_frames[0].shape, "|", label)
                                                                                  
