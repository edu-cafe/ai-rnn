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

"""
## Embedding Layer
* CBOW의 one-hot 입력과 가중치 W_in과의 MatMul 계층의 행렬곱이 수행하는 일은 
  단지 행렬의 특정 행을 추출하는 것뿐임
* 가중치 매게변수로부터 '단어 ID에 해당하는 행(벡터)'을 추출하는 계층을 도입하면됨
* 이 계층을 Embedding Layer라고 함
* Embedding이라는 용어는 Word Embedding이라는 용어에서 유래됨
* 즉, Embedding Layer에 단어 임베딩(분산 표현)을 저장하는 것임  
"""

"""
### 참고
* 자연어 처리 분야에서 단어의 밀집벡터 표현을 ***단어 임베딩(Word Embedding)***
  혹은 단어의 ***분산 표현(Distributed Repersentation)***이라고 함
* 통계 기반 기법으로 얻은 단어 벡터는 Distributional Representation이라고 함
* 신경망을 사용한 추론 기반 기법으로 얻은 단어 벡터는 Distributed Representation이라고 함
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
        out = W[idx]
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
            dW[word_id] += dout[i]
        # or
        # np.add.at(dW, self.idx, dout)  # better
        
        return None

#%%













