# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:44:32 2020

@author: shkim
"""
import numpy as np

"""
## Negative Sampling Layer 구현 : UnigramSampler 클래스 
* UnigramSampler란 이름은 한 단어를 대상으로 확률 분포를 만든다는 의미가 내포됨 
* Unigram이란 '하나의 (연속된) 단어'를 뜻함 
* Bigram : 2개의 연속된 단어, ('you','say'),('you','goodbye')
* Trigram : 3개의 연속된 단어 
"""
import collections
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0  # negative lable로 setting
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, 
                        size=self.sample_size, replace=False, p=p)

        return negative_sample

#%%
"""
## Negative Sampling Layer 구현 : NegativeSamplingLoss 클래스 
"""
import sys
sys.path.append('..')
from myutils.layers import SigmoidWithLoss, EmbeddingDot

class NegativeSamplingLoss:
    # sample_size : negative sample size
    # loss_layers[0] : positive sample layer
    # loss_layer[1]~loss_layer[sample_size] : negative sample layer
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size+1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
    
    # h : hiddel layer unit number
    # target : positive sample target
    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        
        # Positive Sample Forward-Pass
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # Negative Sample Forward-Pass
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i+1].forward(h, negative_target)
            loss += self.loss_layers[i+1].forward(score, negative_label)
        
        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
            
        return dh

#%%
