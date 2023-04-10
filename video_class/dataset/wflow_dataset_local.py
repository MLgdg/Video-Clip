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
import os
import time
import gc
from wflow_sdk.core.dataset import WFlowDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import random
import json
from PIL import Image
import torch
import sys
import librosa
from scipy.io import wavfile
sys.path.append('../')
from dataset.tokenization import BertTokenizer
from utils import _build_vid_pos_ids, MyThread
config = json.load(open('./conf/uvat.json','r'))
import utils
config = utils.Config(config)
import numpy as np

# with open('./data/quert_data_train.json')as ff:
#     for ll in ff:
#         query_nid = json.loads(ll.strip())

class UVATDataset(Dataset):
    def __init__(self, data_file, config, sets='train'):
        super().__init__()
        self.data_file = data_file
        self.video_path =[]
        self.title = []
        self.audio_path = []
        self.label = []
        self.nid = []
        self.num_frames = []
        self.cate = []
        with open(self.data_file)as ff:
            for ll in ff:
                data = json.loads(ll.strip())
                us = data['use']
                if us == sets:
                    self.video_path.append(data['video_path'])
                    self.audio_path.append(data['audio_path'])
                    self.title.append(data['title'])
                    self.label.append(int(data['label']))
                    self.nid.append(data['nid'])
                    self.num_frames.append(data['num_frames'])
                    self.cate.append(data['cate'])
        self.max_lenth = config.max_image
        #self.sets = sets
        #self.label2id = json.load(open(config.label2id, 'r'))
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.tokener = BertTokenizer.from_pretrained('./dataset/')
        #print(self.label2id)
        #print(len(self.data_names))
        self.end_num = max(len(list(set(self.label))),max(self.label)+1)
    def __len__(self):
        return len(self.num_frames)

    def __getitem__(self, i):
        video_path = self.video_path[i]
        audio_paht = self.audio_path[i]
        title = self.title[i]
        label = self.label[i]
        nid = self.nid[i]
        cat = self.cate[i]
        #print(nid)
        #file_name = self.data_names[i]
        #file = self.data.get(file_name)
        #print(file)
        s1= time.time()
        image, image_number = get_mask_video(video_path)
        s2 = time.time()
        audio, audio_len = get_mask_audio(audio_paht)
        s3 = time.time()
        text, text_len = get_mask_title(title)
        s4 = time.time()
        target = torch.tensor(int(label))
        #print('video:{} audio:{}  title:{}'.format(s2-s1,s3-s2,s4-s3))
        image_patch_number = (image.shape[-1] // config.spatial_patch_size) ** 2 * (image_number // config.temporal_patch_size)
        audio_patch_number = audio_len // config.audio_temporal_patch_size #if audio_len>config.audio_temporal_patch_size else 1
        image_attn_masks = torch.ones(image_patch_number, dtype=torch.long)
        audio_attn_masks = torch.ones(audio_patch_number, dtype=torch.long)
        text_attn_masks = torch.ones(text_len, dtype=torch.long)
        return {'video':image, 'video_mask':image_attn_masks, 'audio':audio, 'audio_mask':audio_attn_masks, \
            'title': text, 'title_mask':text_attn_masks, 'tag':target}

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
        lens = [t.shape[-1] for t in tensors]
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
tokener = BertTokenizer.from_pretrained('./dataset')

def get_mask_title(title):
    #print(title)
    if len(title)<1:
        title = "视频无标题"
        text = tokener.tokenize(title)
        text_id = tokener.convert_tokens_to_ids(text)
        len_title = min(config.max_text_len, len(text_id))
        return torch.tensor(text_id[:len_title]), torch.ones(len_title, dtype=torch.long)
    text = tokener.tokenize(title)
    text_id = tokener.convert_tokens_to_ids(text)     
    len_title = min(config.max_text_len, len(text_id))
    return torch.tensor(text_id[:len_title]), len_title
def get_mask_audio(audio_path):
    try:
        _, wav = wavfile.read(audio_path)
        s = time.time()
        wav = wav[: config.max_audio]
        wav = wav/(max(wav)+1e-7)
        audio_mfcc = librosa.feature.mfcc(wav, sr=16000, n_mfcc = 13)
        audio_mfcc = torch.tensor(audio_mfcc).to(torch.float32)
        e = time.time()
        return audio_mfcc, audio_mfcc.shape[-1]
    except Exception as e:
        return torch.Tensor(), 0
transform = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def process_single(img_path, img_names, i):
    img_name = img_names[i]
    image = Image.open(os.path.join(img_path, img_name)).convert('RGB')
    image = transform(image)
    return image

def get_mask_video(video_path):
    imgs = []
    img_names = sorted(os.listdir(video_path))
    num = min(int(len(img_names) // config.temporal_patch_size * config.temporal_patch_size), config.max_image)
    img_names = img_names[:num]
    if num >= config.temporal_patch_size:
        threads = []
        for i in range(num):
            thread = MyThread(process_single, args=(video_path, img_names, i))
            thread.start()
            threads.append(thread)
        for i in range(num):
            thread = threads[i]
            image = thread.get_result()
            imgs.append(image)
    else:
        for img in img_names:
            image = Image.open(os.path.join(video_path, img)).convert('RGB')
            imgs.append(transform(image))
        for i in range(config.temporal_patch_size-num):
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

'''
{'video':image, 'video_mask':image_attn_masks, 'audio':audio, 'audio_mask':audio_attn_masks, 
            'title': text, 'title_mask':text_attn_masks, 'tag':target}
'''

def vqa_collate(feats):
    #print(feats[0]['tag'])
    audio_mfccs = []
    len_audios = []
    videos = [] 
    len_videos = []
    title_ids = []
    title_masks = []
    #query_ids = []
    labels = []
    #query_masks = []
    for feat in feats:
        s1 = time.time()
        audio_mfccs.append(feat.get("audio"))
        len_audios.append(feat.get("audio_mask"))
        videos.append(feat.get("video"))
        len_videos.append(feat.get("video_mask"))
        title_ids.append(feat.get("title"))
        title_masks.append(feat.get("title_mask"))
        #query_ids.extend(feat.get("query_ids"))
        #query_data_labels.extend(feat.get("query_data_labels"))
        #query_masks.extend(feat.get("query_masks"))
        #print(123)
        labels.append(feat.get("tag"))
    s2 = time.time()
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
    s3=time.time()
    labels = torch.stack(labels, dim=0)
    #print(targets)
    s4=time.time()
    img_feat = pad_tensors_img(videos)
    s5=time.time()
    #img_feat = img_feat.permute(0, 2, 1, 3, 4).contiguous()
    if len(img_feat) !=0:
        bs, tmp_image_mask_shape, c, h, w = img_feat.shape
        img_feat = img_feat.permute(0, 2, 1, 3, 4).contiguous()
        image_patch_number = (tmp_image_mask_shape // temporal_patch_size) * ((w // spatial_patch_size) ** 2)
        video_attn_masks = pad_sequence(len_videos, batch_first=True, padding_value=0)
        video_position_ids = _build_vid_pos_ids(tmp_image_mask_shape // temporal_patch_size, w // spatial_patch_size, w // spatial_patch_size).repeat(input_len, 1, 1)
    else:
        video_position_ids = torch.zeros(0, dtype=torch.long)
        video_attn_masks = torch.ones(0, dtype=torch.long)
    s6=time.time()
    audio_feat = pad_tensors_audio(audio_mfccs)
    s7=time.time()
    if len(audio_feat) != 0:
        bs, _, tmp_audio_mask_shape = audio_feat.shape
        audio_patch_number = tmp_audio_mask_shape // audio_temporal_patch_size
        audio_attn_masks_ = pad_sequence(len_audios, batch_first=True, padding_value=0)
        audio_position_ids = torch.arange(0, audio_patch_number, dtype=torch.long).unsqueeze(0).repeat(input_len,1)
    else:
        audio_attn_masks_ = torch.zeros(0, dtype=torch.long)
        audio_position_ids = torch.ones(0, dtype=torch.long)
    s8=time.time()
    #bs, max_tl = input_ids.size()
    #out_size = attn_masks.size(1)
    #gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    e = time.time()
    #print("end vqa_collate")
    gc.collect()
    #print("遍历时间:{},标题处理时间:{},视频对齐处理时间:{}，视频maskdi时间:{} 音频对齐处理时间:{}，音频maskid时间:{}".format(s2-s1,s3-s2,s5-s4,s6-s5,s7-s6,s8-s7))
    batch = {'title_id': title_ids,
             'title_position_id': title_position_ids,
             'title_attn_mask': title_attn_masks, 
             'img_feat': img_feat,
             'video_position_id': video_position_ids,
             'video_attn_mask': video_attn_masks,
             'audio_feat': audio_feat,
             'audio_position_id': audio_position_ids,
             'audio_attn_mask': audio_attn_masks_,
             'target': labels}
    return batch
if __name__ == "__main__":
    #train_dataset = WFlowDataset(name="duanxiao_video_train", version=1)
    #train_dataloader = DataLoader(train_dataset, batch_size=5,collate_fn=vqa_collate, pin_memory=True)
    #print("nids | video_frames_num | video_frames[0].shape | label")
    train_dataset = UVATDataset('/ssd1/gaoqingdong/tdx/fintune/train_test_data_new_cate_v1_0223',config,'test')
    #train_dataloader = DataLoader(train_dataset, batch_size =32, shuffle=True,collate_fn=vqa_collate, num_workers=14)
    print(len(train_dataset ))
    # s = time.time()
    # for ret in train_dataloader:
    #     print(ret.keys())
    #     e = time.time()
    #     print('数据读取时间:', e-s)
    #     s = e
        #print("title_id:",ret['title_id'])
        #print("title_position_id:",ret['title_position_id'])
        #print("title_attn_mask:",ret['title_attn_mask'])
        # print("img_feat shape:",ret['img_feat'].shape)
        # print("video_position_id shape:",ret['video_position_id'].shape)
        # print("video_attn_mask shape:",ret['video_attn_mask'].shape)
        # print("audio_feat shape:",ret['audio_feat'].shape)
        # print("audio_position_id shape:",ret['audio_position_id'].shape)
        # print("audio_attn_mask shape:",ret['audio_attn_mask'].shape)
        # print("target:",ret['target'])
        #break
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
                                                                                  
