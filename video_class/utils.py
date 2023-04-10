'''
@Author: your name
@Date: 2022-01-07 14:29:05
@LastEditTime: 2022-01-14 20:48:39
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /gaoqingdong/图短小融合/model/UVAT/test2/UVAT/utils.py
'''

import os
import threading
import torch
import shutil
class Config:
	def __init__(self, dic):
		self.__dict__.update(dic)

class AverageMeter(object):                                                                                                                                             
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
#        print(pred)
#        print(target.view(1, -1))
        correct = pred.eq(target.view(1, -1).expand_as(pred))
#        print(correct)
        res = []
        
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).data)
        return res

def save_checkpoint(state, is_best, path='./checkpoint', filename='model.pth'):
    if not os.path.exists(path):
        os.makedirs(path)
    full_path = os.path.join(path, filename)
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, os.path.join(path, 'model_best_all_fintunetrian.pth'))
        print("Save best model at %s==" % os.path.join(path, 'model_best.pth'))


def _build_vid_pos_ids(t, h, w, cuda=None):
    """video空间embedding"""
    temporal_ids = torch.arange(t)[:, None, None]  # (t, 1, 1)
    vertical_ids = torch.arange(h)[None, :, None]  # (1, h, 1)
    horizontal_ids = torch.arange(w)[None, None, :]  # (1, 1, w)
    temporal_ids = temporal_ids.repeat(1, h, w)  # (t, h, w)
    vertical_ids = vertical_ids.repeat(t, 1, w)  # (t, h, w)
    horizontal_ids = horizontal_ids.repeat(t, h, 1)  # (t, h, w)
    pos_ids = torch.stack([temporal_ids, vertical_ids, horizontal_ids], dim=3)
    pos_ids = torch.reshape(pos_ids, [1, -1, 3])  # (t*h*w, 3)
    # if cuda is not None:
    #     return pos_ids.cuda(cuda)
    # else:
    return pos_ids
def opt(params):
    """Adam"""
    op = torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-07)
    #op = torch.optim.SGD(params, lr=0.01,momentum=0.9, weight_decay=0.01)
    auto_lr_op = torch.optim.lr_scheduler.ReduceLROnPlateau(op, mode='min', factor =0.9, patience = 20, verbose=False, cooldown = 1)
    return op, auto_lr_op

# MyThread.py线程类
class MyThread(threading.Thread):
    """
    MyThread
    """

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        """
        :return:
        """
        self.result = self.func(*self.args)

    def get_result(self):
        """
        :return:
        """
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None
