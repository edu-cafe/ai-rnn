# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 14:33:03 2020

@author: shkim
"""

"""
## skip-gram 모델
* 하나의 단어로부터 그 주변 단어들을 예측함 
* 따라서 각 출력층에서는 개별적으로 손실을 구하고, 이 개별 손실들을 모두 더한 값을 최종 손실로 함 
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
        self.in_layer = .....
        self.out_layer = .....
        self.loss_layer1 = .....
        self.loss_layer2 = .....
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.______
        s = self.out_layer.______
        l1 = self.loss_layer1.______
        l2 = self.loss_layer2.______
        loss = .....
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
# from myutils.layers import SimpleSkipGram
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












