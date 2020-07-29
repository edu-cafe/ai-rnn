# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:32:57 2020

@author: shkim
"""

"""
# Attention 메카니즘 : seq2seq를 더욱 강력하게 함
* 필요한 정보에만 주목하여 그 정보로부터 시계열 변환을 수행하는 것
* 단어(혹은 문구)의 대응관계를 나타내는 정보를 **얼라인먼트(alignment)**라고 함
* 지금까지는 alignment를 주로 사람이 수작업으로 만들었으나,
* Attention 기술은 alignment라는 아이디어를 seq2seq에 자동으로 도입하는 데 성공함

## seq2seq의 문제점
* seq2seq에서는 Encoder가 시계열 데이터를 인코딩하고, 인코딩된 정보를 Decoder로 전달함
* 이때 Encoder의 출력은 **'고정 길이의 벡터'**임
* '고정 길이'라는 데는 큰 문제가 잠재해 있음
* 고정 길이 벡터라 함은 입력 문장의 길이에 관계없이(아무리 길어도) 항상 같은 길이의 벡터로 변환한다는 뜻임
* 즉 아무리 긴 문장이 입력되더라도 항상 똑같은 길이의 벡터에 밀어 넣어야 함
* 즉 기존 seq2seq Encoder는 아무리 긴 문장이라도 고정 길이의 벡터로 변환함
  --> 필요한 정보가 벡터에 다 담기지 못하게 됨

## Encoder 개선
* Encoder 출력의 길이는 입력 문장의 길이에 따라 바꿔주는 것이 개선의 포인트임
* 시각별 LSTM 계층의 은닉 상태 벡터를 모두 이용하는 것임
* 각 시각(단어)의 은닉 상태 벡터를 모두 이용하면 입력된 단어와 같은 수의 벡터를 얻을 수 있음
* 각 시각의 은닉 상태에는 직전에 입력된 단어에 대한 정보가 많이 포함되어 있음
* Encoder가 출력하는 hs 행렬은 각 단어에 해당하는 벡터들의 집합이라고 볼 수 있음

## Decoder 개선
* 개선 -1 : 각 단어의 중요도를 나타내는 가중치 a와 각 단어의 벡터 hs로부터 
  가중합(weighted sum)을 구하여 '맥락 벡터'를 얻음
* 개선 -2 : 벡터의 '내적'을 이용하여 가중치 a를 구함
* 개선 -3 : AttentionWeight과 WeightSum을 연결하여 맥락 벡터 c를 완성함

* myutils.attention_layer
"""

import numpy as np
import sys
sys.path.append('..')
from myutils.layers import Softmax

#%%
"""
## WeightSum 계층 구현 
"""
class WeightSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        
    # a : alignment(weight)
    def forward(self, hs, a):
        N, T, H = hs.shape
        
        ar = a.reshape(N, T, 1).repeat(H, axis=2)  # (N, T, H)
        t = hs * ar  # (N, T, H)
        
        c = np.sum(t, axis=1)  # (N, H), context vector
        
        self.cache = (hs, ar)
        return c  # context vector
    
    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)  # sum의 역전파 (N, T, H)
        dar = dt * hs  # (N, T, H)
        dhs = dt * ar # (N, T, H)
        da = np.sum(dar, axis=2)  # (T, H)
        
        return dhs, da
        
#%%
"""
* 가중치 a(alignment, attention value)를 구하는 클래스 구현 : AttentionWeight
"""
class AttentionWeight:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None
        
    def forward(self, hs, h):
        N, T, H = hs.shape
        
        hr = h.reshape(N, 1, H).repeat(T, axis=1)  # (N, T, H)
        # hr = a.reshape(N, 1, H)  # using broadcasting
        t = hs * hr  # (N, T, H)
        s = np.sum(t, axis=2)  # (N, T)
        a = self.softmax.forward(s)  # (N, T) alignment(weight)
        
        self.cache = (hs, hr)
        return a
    
    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)  # (N, T, H)
        dhs = dt * hr  # (N, T, H)
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)  # (N, H)

        return dhs, dh
        
#%%
"""
*  AttentionWeight과 WeightSum을 연결하는 클래스 구현 : Attention 
"""
class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None
        
    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)  # alignment(weight)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out
    
    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh

#%%
"""
* 다수의 Attention 계층을 내포하는 클래스 구현 : TimeAttention 
"""
class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weight = None  # (N, 1, T)
        
    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weight = []
        
        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weight.append(layer.attention_weight)

        return out
    
    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)
        
        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh
        
        return dhs_enc, dhs_dec
        
        
        
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh    

#%%




