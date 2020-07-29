# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:23:02 2020

@author: shkim
"""

import numpy as np
import sys
sys.path.append('..')
from myutils.layers import Embedding
from myutils.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    # 맥락 단어의 크기 : window_size * 2
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        
        # Weight Initialize
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')
        
        # Layer Create
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
        
        # Aggregate all Weights and Gradients
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # Save word distributed representation (word vectors)
        self.word_vecs = W_in
    
    # contexts : 맥락 단어 ID, 2차원 배열 
    # target : 타깃 단어 ID, 1차원 배열 
    def forward(self, context, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(context[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

#%%

