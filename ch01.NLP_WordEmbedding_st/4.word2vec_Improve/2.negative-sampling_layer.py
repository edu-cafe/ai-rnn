# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:36:33 2020

@author: shkim
"""

"""
## 은닉층 이후에 계산이 오래 걸리는 곳
* 은닉층 뉴런과 가중치 행렬(W_out)의 곱
* Softmax 계층의 계산
"""

"""
## Negative Sampling의 도입 
* word2vec에서의 은닉층 이후의 행렬곱과 Softmax 계층 계산 병목 개선 
* Softmax 대신 Negative Sampling을 이용하면 어휘가 아무리 많아져도 계산량을 
  낮은 수준에서 일정하게 억제할 수 있음
"""

"""
## Negative Sampling 이란?
* 이 기법의 핵심 아이디어는 ***이진 분류(binary classification)***에 있음 
* 즉, 다중 분류(multi-class clasiification)를 이진 분류로 근사하는 것임 
"""

#%%
"""
## EmbeddingDot Layer 구현
"""
import sys
sys.path.append('../../')
from myutils.layers import Embedding
import numpy as np

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
    
    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
    
#%%
"""
## Negative Sampling 기법 개요 
* Negative Sampling 기법은 적은 수의 부정적 예를 샘플링해서 사용함
* Negative Sampling 기법은 긍정적 예를 타깃으로 한 경우의 손실을 구함
* 그와 동시에 부정적인 예를 몇 개 샘플링(선별)하여 그 부정적 예에 대해서도 마찬가지로 손실을 구함
* 그리고 각각의 데이터(긍정적 예와 부정적 예)의 손실을 더한 값을 최종 손실로 함
"""

#%%
"""
## Negative Sampling의 샘플링 기법
* 말뭉치에서의 단어 빈도를 기준으로 샘플링함 
* 말뭉치에서 각 단어의 출현 횟수를 구해 ***'확률 분포'***로 나타냄 
* 그런 다음, 그 확률 분포대로 샘플링함 
* np.random.choice() 사용
"""

import numpy as np

# 0~9 숫자 중 하나를 무작위로 샘플링
print(np.random.choice(10))

# words에서 하나만 무작위로 샘플링
words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
print(np.random.choice(words))

#%%
# words에서 5개만 무작위로 샘플링(중복 있음)
print(np.random.choice(words, size=5))

# words에서 5개만 무작위로 샘플링(중복 없음)
print(np.random.choice(words, size=5, replace=False))

#%%
# 확률 분포에 따라 샘플링
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
print(np.random.choice(words, p=p))

print(np.random.choice(words, p=p, size=5))
print(np.random.choice(words, p=p, size=5, replace=False))

#%%
"""
### 확률이 낮은 단어를 버리지 않기 위한 방법
* 각 단어가 발갱할 확률에 0.75를 곱하므로써 원래 확률이 낮은 단어의 확률을 살짝 올릴 수 있음
"""
p = [0.7, 0.29, 0.01]
new_p = np.power(p, 0.75)
new_p /= np.sum(new_p)
print(new_p)

"""
### Negative Sampling 방법 정리 
* Negative Sampling은 말뭉치에서 단어의 확률 분포를 만들고,
* 다시 0.75를 제곱한 다음
* np.random.choice()를 사용해 부정적 예를 샘플링함  
"""

#%%
"""
## Negative Sampling Layer 구현 : UnigramSampler 클래스 
* UnigramSampler란 이름은 한 단어를 대상으로 확률 분포를 만든다는 의미가 내포됨 
* Unigram이란 '하나의 (연속된) 단어'를 뜻함 
* Bigram : 2개의 연속된 단어, ('you','say'),('you','goodbye')
* Trigram : 3개의 연속된 단어 
"""
import collections
class UnigramSampler:
    # sample_size : negative sample의 개수
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

    # target당 sample_size 만큼의 negative sample을 추출함
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
corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus, power, sample_size)
target = np.array([1, 3, 0])  # positive target 3개
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)

#%%
"""
## Negative Sampling Layer 구현 : NegativeSamplingLoss 클래스 
"""
import sys
sys.path.append('../../')
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














