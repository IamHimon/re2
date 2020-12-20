# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as f

"""
定义自已的网络层：
1. 构建类，继承nn.Module
2. 实现构造函数 __init__(),定义一些属性。一般把网络中具有可学习参数的层放在构造函数。
3. 实现forward()方法，执行前向计算。
    forward()方法接受一个输入, 同时必须一个输出。forward方法中可以调用其他modules也可以执行任意的Tensor计算。
    然后通过其他modules或者其他Function运算，来进行forward，返回一个输出结果。
    只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
    执行方法：embedding = Embedding(args) -> embedding(tensor) -> 执行__call__(), 默认register_forward_hook() -> forward(tensor)
    
"""


class Embedding(nn.Module):
    def __init__(self, args):
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super().__init__()
        self.fix_embeddings = args.fix_embeddings
        self.embedding = nn.Embedding(args.num_vocab, args.embedding_dim, padding_idx=0)
        self.dropout = args.dropout

    def set_(self, value):
        self.embedding.weight.requires_grad = not self.fix_embeddings
        self.embedding.load_state_dict({'weight': torch.tensor(value)})

    def forward(self, x):
        x = self.embedding(x)
        x = f.dropout(x, self.dropout, self.training)
        return x
