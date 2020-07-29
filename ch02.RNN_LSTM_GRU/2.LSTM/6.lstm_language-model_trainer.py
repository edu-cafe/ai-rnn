# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:02:40 2020

@author: shkim
"""

"""
## LSTM Language Model Trainner 클래스 구현
* myutils.trainer.RnnlmTrainer
"""

#%%
"""
## LSTM Language Model Trainner 클래스를 이용한 언어 모델 Training 코드 구현
* PTB 데이터셋을 이용하여 LSTM 언어 모델을 학습시켜보자 

## RnnlmTrainer의 모델 Training 순서
* 1) 미니배치를 순차적으로 만들어
* 2) 모델의 순전파와 역전파를 호출하고
* 3) 옵티마이저로 가중치를 갱신하고
* 4) perplexity를 구함
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.optimizer import SGD
from ptb_dataset import load_data
from myutils.time_layers import LstmLm
from myutils.trainer import RnnlmTrainer

#%%
# Setting Hyperparameters
batch_size = 20
wordvec_size = 100
hidden_size = 100  # LSTM 은닉 상태 벡터의 원소 수
time_size = 35  # Truncated BPTT가 한번에 펼치는 시간 크기 
lr = 20.0
max_epoch = 4
max_grad = 0.25  # for Gradients Clipping

# Read Taining Data
corpus, word_to_id, id_to_word = load_data('train')
corpus_test, _, _ = load_data('test')
vocab_size = len(word_to_id)

xs = corpus[:-1]  # input
ts = corpus[1:]  # label
data_size = len(xs)
print('말뭉치 크기:%d, 어휘수:%d, 입력 데이터 크기:%d' % 
      (len(corpus), vocab_size, data_size))  # 929589, 10000, 929588

#%%
# Create Model
model = _____(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = ______(model, optimizer)

# Gradients Clipping을 적용하여 학습 
# eval_interval : perplexity evaluation 주기 
trainer.____(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)

"""
| 에폭 1 |  반복 1 / 1327 | 시간 0[s] | 퍼플렉서티 10000.63
| 에폭 1 |  반복 21 / 1327 | 시간 5[s] | 퍼플렉서티 2925.99
| 에폭 1 |  반복 41 / 1327 | 시간 11[s] | 퍼플렉서티 1252.64
| 에폭 1 |  반복 61 / 1327 | 시간 17[s] | 퍼플렉서티 995.25
| 에폭 1 |  반복 81 / 1327 | 시간 23[s] | 퍼플렉서티 814.08
| 에폭 1 |  반복 101 / 1327 | 시간 30[s] | 퍼플렉서티 665.74
         :
| 에폭 4 |  반복 1221 / 1327 | 시간 1670[s] | 퍼플렉서티 76.56
| 에폭 4 |  반복 1241 / 1327 | 시간 1677[s] | 퍼플렉서티 91.17
| 에폭 4 |  반복 1261 / 1327 | 시간 1683[s] | 퍼플렉서티 94.10
| 에폭 4 |  반복 1281 / 1327 | 시간 1690[s] | 퍼플렉서티 89.93
| 에폭 4 |  반복 1301 / 1327 | 시간 1696[s] | 퍼플렉서티 111.71
| 에폭 4 |  반복 1321 / 1327 | 시간 1703[s] | 퍼플렉서티 111.79
"""
#%%
trainer.plot()

#%%
# test 데이터로 평가
from myutils.util import eval_perplexity

model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test data perplexity: ', ppl_test)  # 137.95755965048963

#%%
# 매개변수 저장 
model.save_params()  # LstmLm.pkl

#%%