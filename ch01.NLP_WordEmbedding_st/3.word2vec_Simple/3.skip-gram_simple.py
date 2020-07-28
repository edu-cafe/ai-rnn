# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:33:03 2020

@author: shkim
"""

"""
## CBOW 모델과 확률
* 확률 P(A) : A라는 현상이 일어날 확률
* 동시확률 P(A,B) : A와 B가 동시에 일어날 확률
* 사후확률 P(A|B) : 사건이 일어난 후의 확률, B(라는 정보)가 주어졌을 때 A가 일어날 확률 

## 맥락으로 W_t-1과 W_t+1이 주어졌을 때 타깃이 W_t가 될 확률을 수깃으로 표현하면?
* P(W_t | W_t-1, W_t+1)
* W_t-1과 W_t+1이 일어난 후 W_t가 일어날 확률, W_t-1과 W_t+1이 주어졌을 때 W_t가 일어날 확률
* 이는 CBOW의 모델링임 

## CBOW 모델의 손실 함수(Loss Function)
* L = -logP(W_t | W_t-1, W_t+1)  //음의 로그 가능도(Negative Log Likelihood)
* 위 Loss를 말뭉치 전체로 확장하면 다음과 같음
  L = -(1/T)SUM_t=1~T(logP(W_t | W_t-1, W_t+1))
"""

#%%
"""
## skip-gram 모델
* 하나의 단어로부터 그 주변 단어들을 예측함 
* skip-gram 모델은 CBOW에서 다루는 맥락과 타깃을 역전시킨 모델임 Fig.3-23
* CBOW 모델은 맥락이 여러 개 있고, 그 여러 맥락으로부터 중앙의 단어(타깃)을 추측함
* skip-gram 모델은 중앙의 단어(타깃)로부터 주변의 여러 단어(맥락)를 추측함 
* skip-gram 모델의 입력층은 하나이고 출력층은 맥락의 수만큼 존재함 
* 따라서 각 출력층에서는 개별적으로 손실을 구하고, 이 개별 손실들을 모두 더한 값을 최종 손실로 함 

## skip-gram 모델을 확률 표기로 나타내 보자
* P(W_t-1, W_t+1 | W_t)   //W_t가 주어졌을 때 W_t-1과 W_t+1이 동시에 일어날 확률
* 맥락들의 단어들이 '조건부 독립'이라고(맥락의 단어들 사이에 관련성이 없다고) 가정하면 다음과 같이 변환 가능함
  P(W_t-1, W_t+1 | W_t) = P(W_t-1 | W_t) * P(W_t+1 | W_t) 
  
## skip-gram 모델의 손실 함수(Loss Function)
* L = -logP(W_t-1, W_t+1 | W_t)
* L = -logP(W_t-1 | W_t)*P(W_t+1 | W_t)
* L = -(logP(W_t-1 | W_t) + logP(W_t+1 | W_t))

* 위 Loss를 말뭉치 전체로 확장하면 다음과 같음
  L = -(1/T)SUM_t=1~T(logP(W_t-1 | W_t) + logP(W_t+1 | W_t))
"""

#%%
import sys
sys.path.append('../../')
import numpy as np
from myutils.layers import MatMul, SoftmaxWithLoss

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, contexts[:, 0])
        l2 = self.loss_layer2.forward(s, contexts[:, 1])
        loss = l1 + l2
        return loss

    def backward(self, dout=1):
        dl1 = self.loss_layer1.backward(dout)
        dl2 = self.loss_layer2.backward(dout)
        ds = dl1 + dl2
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None

#%%
"""
## skip-gram 모델 학습(training) 코드 구현 
*
"""

import sys
sys.path.append('../../')
from myutils.util import preprocess, create_contexts_target, convert_one_hot
from myutils.layers import SimpleSkipGram
from myutils.optimizer import Adam
from myutils.trainer import Trainer

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
print(target)  # [1 2 3 4 1 5]
print(contexts)  # [[0 2] [1 3] [2 4] [3 1] [4 5] [1 6]]

#%%
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
print('target_shape:', target.shape)  # (6, 7)
print('contexts_shape:', contexts.shape)  # (6, 2, 7)

#%%
model = SimpleSkipGram(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)

#%%
trainer.plot()

#%%
"""
## 학습이 끝난 후의 가중치 매개변수 출력 
* 입력 측 MatMul 계층의 가중치는 instance 변수 word_vecs에 저장되어 있음 
* word_vecs의 각 행에는 대응하는 단어 ID의 분산 표현이 저장되어 있음 
* 단어를 밀집벡터(분산 표현)로 나타낼 수 있음 : 단어의 의미를 잘 파악한 벡터 표현 
"""

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():    
    print(word, word_vecs[word_id])

"""
you [-0.00410732  0.00791402 -0.02264722  0.0066872  -0.00568283]
say [-0.6857684   0.85931647  0.8997579  -0.884531   -0.8731836 ]
goodbye [ 1.3790207  -0.79659003 -0.7837708   0.77834535  0.806431  ]
and [ 1.3305069   0.9508089   0.9267044  -0.92633337 -0.94237316]
i [ 1.384568   -0.802843   -0.79307777  0.78893954  0.7897504 ]
hello [-1.1972597  -0.8836372  -0.87943333  0.8995975   0.8721954 ]
. [-0.0211845   0.01583714 -0.00408299 -0.0050106  -0.00177043]
"""

#%%
"""
## word2vec 모델 : CBOW vs. skip-gram
* 단어의 분산 표현의 정밀도 면에서 skip-gram 모델의 결과가 더 좋은 경우가 많음
* 특히 말뭉치가 커질수록 저빈도 단어나 유추 문제의 성능 면에서 skip-gram 모델이 더 뛰어난 경향이 있음
* 학습 속도 면에서는 CBOW 모델이 더 빠름
* skip-gram 모델은 손실을 맥락의 수만큼 구해야 해서 계산 비용이 그만큼 커짐
* skip-gram 모델은 하나의 단어로부터 그 주변 단어들을 례측하기 때문에 꽤 어려운 문제임
* skip-gram이 더 어려운 문제에 도전한다고 할 수 있음
* skip-gram은 더 어려운 상황에서 단련하는 만큼 이 모델이 내어 주는 단어의 분산 표현이 더 뛰어날 가능성이 커짐
"""

#%%












