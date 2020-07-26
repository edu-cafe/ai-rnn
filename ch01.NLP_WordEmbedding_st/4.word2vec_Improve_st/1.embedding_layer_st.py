# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:22:11 2020

@author: shkim
"""

"""
## CBOW 모델의 문제점
* CBOW 모델도 작은 말뭉치에서는 특별히 문제될 것이 없음
* 어휘가 100만개, 은닉층의 뉴런이 100개인 CBOW 모델인 경우 다음의 두 계산 병목 현상이 발생함
* 입력층의 원핫 표현과 가중치 행렬 W_in의 곱 계산 --> Embedding Layer로 해결
* 은닉층의 가중치 행렬 W_out의 곱 및 Softmax 계층의 계산 --> Negative Sampleing으로 해결
"""

#%%
"""
## Embedding Layer 구현

"""
#%%
import numpy as np

W = np.arange(21).reshape(7, 3)
print(W)
print(W[1])
print(W[3])

#%%
idx = np.array([1, 0, 3, 1])  # mini-batch idx
print(W[idx])

#%%
W = np.arange(21).reshape(7, 3)
print(W)
W[...] = 0
print(W)

#%%
class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = ......
        return out
    
    # def backward(self, dout):
    #     dW, = self.grads
    #     dW[...] = 0
    #     dW[self.idx] = dout  # bad ex --> change later
    #     return None
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        
        for i, word_id in enumerate(self.idx):
            dW[word_id] += .....
        # or
        # np.add.at(dW, self.idx, dout)  # better
        
        return None

#%%













