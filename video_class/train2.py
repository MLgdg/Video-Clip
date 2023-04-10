'''
@Author: your name
@Date: 2022-01-06 20:29:21
@LastEditTime: 2022-01-14 20:48:46
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /gaoqingdong/图短小融合/model/UVAT/test2/UVAT/train2.py
'''
from dataset import gendata
import json
import argparse
#from wflow_dataloader import WFlowDataLoader as DataLoader
from torch.utils.data import DataLoader
import utils
import torch
import numpy as np
import torch.nn as nn
from wflow_sdk.core.dataset import WFlowDataset
#from dataset import gendata
import time
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

#from dataset import feature_gendata
#from dataset import video_gendata
from dataset import wflow_dataset_local
from model import uvat
config = json.load(open('./conf/uvat.json','r'))
config = utils.Config(config)
def no_fn(input):
    return input

def main(args):
        #device = torch.device('cpu')
    #datas = gendata.UVATDataset('./data/UVAT_train_only_has_video_title.json', config)
    train_dataset = wflow_dataset_local.UVATDataset('/home/bpfsrw3/gaoqingdong/tdx_fintune/train_test_data_new_cate_v1_0223',config,'train')
    train_dataloader = DataLoader(train_dataset, batch_size =args.batchsize, shuffle=True,collate_fn=wflow_dataset_local.vqa_collate, num_workers=50)
    #train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=wflow_dataset.collate0, num_workers=32, pin_memory=False, num_data_processers=8, data_processer=wflow_dataset.vqa_collate, data_processer_batch_size=args.batchsize)
    test_dataset = wflow_dataset_local.UVATDataset('/home/bpfsrw3/gaoqingdong/tdx_fintune/train_test_data_new_cate_v1_0223',config,'test')
    test_dataloader = DataLoader(test_dataset, batch_size =args.batchsize, shuffle=False,collate_fn=wflow_dataset_local.vqa_collate, num_workers=50)
    #test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=wflow_dataset.collate0, num_workers=32, pin_memory=False, num_data_processers=8, data_processer=wflow_dataset.vqa_collate, data_processer_batch_size=args.batchsize)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize,collate_fn=wflow_dataset.vqa_collate, pin_memory=False, num_workers=2)
    #datas = feature_gendata.UVATDataset('./data/UVAT_train_categroy.json', config)
    #dataloade = DataLoader(datas, args.batchsize, shuffle=True, collate_fn=feature_gendata.vqa_collate)
    end_num = max(train_dataset.end_num,test_dataset.end_num)
    if torch.cuda.is_available() and args.gpu:
        device = torch.device("cuda:{}".format(args.CUDA[0]))
        fc = torch.nn.Linear(config.hidden_size, end_num).to(device)
    else:
        device = torch.device('cpu')
        fc = torch.nn.Linear(config.hidden_size, end_num)

    Umodel = uvat.UniterModel(config).to(device)
    if len(args.CUDA)>1: 
        Umodel=nn.DataParallel(Umodel,device_ids=args.CUDA)
    #print('多GPU')
    optim, auto_lr_op = utils.opt(Umodel.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    is_best = 1
    best_prec = 0
    best_loss = 100
    start_epoch = 0
    start_data_loader_time = 0
    if args.RESUME:
        path_checkpoint = "./{}/m.pth".format(args.checkpoint)  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        Umodel.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        #print("加载断点模型")
        #optim.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        #start_epoch = checkpoint['epoch']  # 设置开始的epoch
    #if len(args.CUDA)>1:
    #    Umodel.module.fc = 
    Umodel.module.fc = fc
    for e in range(start_epoch, args.epoch):
        loss_assess = utils.AverageMeter()
        acc1_assess = utils.AverageMeter()
        acc5_assess = utils.AverageMeter()
        printf_num = 0 
        s1 = time.time()
        for bn, batch in enumerate(train_dataloader):
            # print(ret.keys())
            # print("title_id:",ret['title_id'])
            # print("title_position_id:",ret['title_position_id'])
            # print("title_attn_mask:",ret['title_attn_mask'])
            # print("img_feat shape:",ret['img_feat'].shape)
            # print("video_position_id shape:",ret['video_position_id'].shape)
            # print("video_attn_mask shape:",ret['video_attn_mask'].shape)
            # print("audio_feat shape:",ret['audio_feat'].shape)
            # print("audio_position_id shape:",ret['audio_position_id'].shape)
            # print("audio_attn_mask shape:",ret['audio_attn_mask'].shape)
            # print("target:",ret['target'])
            # break
    #     # break
            s2=time.time()
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = Umodel(batch["img_feat"], batch["video_position_id"], batch["video_attn_mask"],\
                        batch["audio_feat"],batch["audio_position_id"],batch["audio_attn_mask"],\
                        batch["title_id"],batch['title_position_id'], batch['title_attn_mask']\
                        )
            s3 = time.time()
            #print(outputs.shape)
            #print("数据处理时间:",s2-s1,"模型forword时间",s3-s2)
            #s1=s2
            optim.zero_grad()
            #s4 = time.time()
            #print(torch.max(batch["target"]))
            loss = loss_fn(outputs, batch["target"])
            loss.backward()
            optim.step()
            s5= time.time()
            #print(len(batch["target"]))
            #print("模型back时间",s5-s4)
            loss_assess.update(loss.cpu().data.numpy(), len(batch["target"]))
            acc = utils.accuracy(outputs.cpu(), batch["target"].cpu())
            acc1_assess.update(acc[0][0], len(batch["target"]))
            acc5_assess.update(acc[1][0], len(batch["target"]))
            if printf_num % args.printf == 0:
                print("epoch:{}--batch:{}/{}--batchloss:{:.4f}--batchacc:{:.4f}--avgloss:{:.4f}--avgaccTop1:{:.4f}--avgaccTop5:{:.4f}--data_time:{:.4f}---model_time:{:.4f}".format(
                    e, bn,len(train_dataloader), loss.cpu().data.numpy(), acc[0][0], loss_assess.avg, acc1_assess.avg,acc5_assess.avg,s2-s1,s5-s2))
            s1=s2

            #start_data_loader_time = time.time()
            if (bn+1) % args.lr_check_f == 0:
                with torch.no_grad():
                    test_loss_assess = utils.AverageMeter()
                    test_acc1_assess = utils.AverageMeter()
                    test_acc5_assess = utils.AverageMeter()
                    for bn, batch in enumerate(test_dataloader):
                        batch = {key: value.to(device) for key, value in batch.items()}
                        outputs = Umodel(batch["img_feat"], batch["video_position_id"], batch["video_attn_mask"],\
                                    batch["audio_feat"],batch["audio_position_id"],batch["audio_attn_mask"],\
                                    batch["title_id"],batch['title_position_id'], batch['title_attn_mask']
                                    )  
                        loss = loss_fn(outputs, batch["target"])                      
                        test_loss_assess.update(loss.cpu().data.numpy(), len(batch["target"]))
                        test_acc = utils.accuracy(outputs.cpu(), batch["target"].cpu())
                        test_acc1_assess.update(test_acc[0][0], len(batch["target"]))
                        test_acc5_assess.update(test_acc[1][0], len(batch["target"]))
                        print("epoch:{}--batch:{}/{}--testbatchloss:{:.4f}--testbatchacc:{:.4f}--testavgloss:{:.4f}--testavgaccTop1:{:.4f}----testavgaccTop5:{:.4f}".format(
                            e, bn,len(test_dataloader), loss.cpu().data.numpy(), test_acc[0][0], test_loss_assess.avg, test_acc1_assess.avg,test_acc5_assess.avg)) 
                old_lr = optim.param_groups[0]['lr']
                auto_lr_op.step(test_loss_assess.avg)
                new_lr = optim.param_groups[0]['lr']
                if old_lr != new_lr:
                    print("自动调整学习 lr from {} to {}".format(old_lr, new_lr))
                is_best = test_acc1_assess.avg > best_prec
                utils.save_checkpoint({'net': Umodel.state_dict(),'optimizer':optim.state_dict(),'epoch':e }, is_best, path=args.checkpoint)
                best_loss = max(test_acc1_assess.avg, best_loss)
            printf_num += 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=10, type=int, help="epoch size")
    parser.add_argument("--printf", default=2, type=int, help="打印日志频率")
    parser.add_argument("--CUDA", default=0, nargs='+', type=int, help="GPU")
    parser.add_argument("--gpu", default=1, type=int, help="use gpu")
    parser.add_argument("--batchsize", default=16, type=int, help="batch size")
    parser.add_argument("--lr_check_f", default=512, type=float, help="auto check lr and save model f")
    parser.add_argument("--checkpoint", default='checkpoint', type=str, help="best model directory ")
    parser.add_argument("--RESUME", default='0', type=int, help="断点续训 ")
    parser.add_argument("--num_workers", default=8, type=int, help="num workers ")
    args = parser.parse_args()
    main(args)
