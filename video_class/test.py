# -*- coding: utf-8 -*-
"""
#####################################################################
    > File Name: test.py
    > Author: 
    > Email: @baidu.com
    > Created Time: 2022/01/30 15:46:35
    > Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#####################################################################
"""
import sys
import torch
import torch.nn as nn

x = torch.tensor([5.0, -1.0], dtype=torch.float).cuda(5).view(-1, 1)

#model = nn.Linear(in_features=1, out_features=1, bias=False).cuda()
model = nn.parallel.DataParallel(nn.Linear(in_features=1, out_features=1, bias=False).cuda(5), device_ids=[5,7])

y = model(x)
print(y.shape)

label = torch.zeros(2, 1, dtype=torch.float).cuda(5)

loss = torch.sum((y - label)**2)

loss.backward()
print(loss)
