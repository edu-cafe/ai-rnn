# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:12:45 2020

@author: shkim
"""

"""
## RNN Language Model Trainner 클래스 구현
* myutils.trainer.RnnlmTrainer
"""

#%%
"""
## RNN Language Model Trainner 클래스를 이용한 언어 모델 Training 코드 구현
* PTB 데이터셋을 이용하여 RNN 언어 모델을 학습시켜보자 
* PTB 데이터셋의 일부(1000개 단어)만 사용할 것임 
  --> 아직 성능이 나오지 않은 모델이기 때문 --> 다음 코드에서 개선할 것임(LSTM, GRU)

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
from myutils.time_layers import SimpleRnnLm
from myutils.trainer import RnnlmTrainer

#%%
# Setting Hyperparameters
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN 은닉 상태 벡터의 원소 수
time_size = 5  # Truncated BPTT가 한번에 펼치는 시간 크기 
lr = 0.1
max_epoch = 100

# Read Taining Data : 전체 데이터 중 1000개
corpus, word_to_id, id_to_word = load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
# print('corpus:', corpus[500:510])  # [241 242  42  61  26 243 108 244 172  48]
# print('vocab_size:', vocab_size)  # 418

xs = corpus[:-1]  # input
ts = corpus[1:]  # label
data_size = len(xs)
print('말뭉치 크기:%d, 어휘수:%d, 입력 데이터 크기:%d' % 
      (corpus_size, vocab_size, data_size))  # 1000, 418, 999

#%%
# Variables for training
max_iters = data_size // (batch_size * time_size)  # 19
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []  # perplexity list

# Create Model
model = SimpleRnnLm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)
trainer.fit(xs, ts, max_epoch, batch_size, time_size)

"""
| 에폭 1 |  반복 1 / 19 | 시간 0[s] | 퍼플렉서티 417.50
| 에폭 2 |  반복 1 / 19 | 시간 0[s] | 퍼플렉서티 369.04
| 에폭 3 |  반복 1 / 19 | 시간 0[s] | 퍼플렉서티 253.32
| 에폭 4 |  반복 1 / 19 | 시간 0[s] | 퍼플렉서티 219.36
| 에폭 5 |  반복 1 / 19 | 시간 0[s] | 퍼플렉서티 210.31
         :
| 에폭 96 |  반복 1 / 19 | 시간 3[s] | 퍼플렉서티 7.39
| 에폭 97 |  반복 1 / 19 | 시간 3[s] | 퍼플렉서티 7.45
| 에폭 98 |  반복 1 / 19 | 시간 4[s] | 퍼플렉서티 6.93
| 에폭 99 |  반복 1 / 19 | 시간 4[s] | 퍼플렉서티 6.44
| 에폭 100 |  반복 1 / 19 | 시간 4[s] | 퍼플렉서티 6.41
"""
#%%
trainer.plot()

#%%




