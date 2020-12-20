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
from .modules import Module, ModuleList, ModuleDict
from .modules.embedding import Embedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import Pooling
from .modules.prediction import registry as prediction


"""
模型
"""


class Network(Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding = Embedding(args)
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(args, args.embedding_dim if i == 0 else args.embedding_dim + args.hidden_size),
            'alignment': alignment[args.alignment](
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
            'fusion': fusion[args.fusion](
                args, args.embedding_dim + args.hidden_size if i == 0 else args.embedding_dim + args.hidden_size * 2),
        }) for i in range(args.blocks)])
        self.connection = connection[args.connection]()
        self.pooling = Pooling()
        self.prediction = prediction[args.prediction](args)

    def forward(self, inputs):
        a = inputs['text1']     # (batch_size, len_a)
        b = inputs['text2']     # (batch_size, len_b)
        mask_a = inputs['mask1']
        mask_b = inputs['mask2']

        a = self.embedding(a)   # (batch_size,len_a,glove_dim)
        b = self.embedding(b)   # (batch_size,len_b,glove_dim)
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a) # (b_s,len_a,hidden_size)
            b_enc = block['encoder'](b, mask_b) # (b_s,len_b,hidden_size)
            a = torch.cat([a, a_enc], dim=-1)   # (b_s,len_a,glove_dim +hidden_size)
            b = torch.cat([b, b_enc], dim=-1)   # (b_s,len_b,glove_dim +hidden_size)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        return self.prediction(a, b)
