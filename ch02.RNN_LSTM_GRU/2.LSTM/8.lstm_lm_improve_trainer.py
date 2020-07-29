# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 21:55:55 2020

@author: shkim
"""

"""
## 개선된 LSTM Language Model Training 코드 구현

* myutils.time_layers --> BetterLstmLm Class
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.optimizer import SGD
from myutils.trainer import RnnlmTrainer
from myutils.util import eval_perplexity
from myutils.time_layers import BetterLstmLm
from ptb_dataset import load_data

#%%
# Setting Hyperparameters
batch_size = 20
wordvec_size = 650
hidden_size = 650  # LSTM 은닉 상태 벡터의 원소 수
time_size = 35  # Truncated BPTT가 한번에 펼치는 시간 크기 
lr = 20.0  
max_epoch = 10  # 12hr
max_grad = 0.25
dropout = 0.5

# Read Taining Data
corpus, word_to_id, id_to_word = load_data('train')
corpus_val, _, _ = load_data('val')
corpus_test, _, _ = load_data('test')
vocab_size = len(word_to_id)

xs = corpus[:-1]  # input
ts = corpus[1:]  # label
data_size = len(xs)
print('말뭉치 크기:%d, 어휘수:%d, 입력 데이터 크기:%d' % 
      (len(corpus), vocab_size, data_size))  # 929589, 10000, 929588

#%%
# Create Model
model = BetterLstmLm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

#%%
best_ppl = float('inf')
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, 
                time_size=time_size, max_grad=max_grad)
    
    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print('-->perplexity: ', ppl)
    
    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()  # 매개변수 저장 
    else:
        lr /= 4.0
        optimizer.lr = lr
        
    model.reset_state()
    print('-'*50)

"""
| 에폭 4 |  반복 1281 / 1327 | 시간 3572[s] | 퍼플렉서티 103.01
| 에폭 4 |  반복 1301 / 1327 | 시간 3628[s] | 퍼플렉서티 130.16
| 에폭 4 |  반복 1321 / 1327 | 시간 3683[s] | 퍼플렉서티 128.11
퍼플렉서티 평가 중 ...
209 / 210
-->perplexity:  112.3351335172994
--------------------------------------------------
| 에폭 5 |  반복 1261 / 1327 | 시간 3503[s] | 퍼플렉서티 93.01
| 에폭 5 |  반복 1281 / 1327 | 시간 3558[s] | 퍼플렉서티 94.92
| 에폭 5 |  반복 1301 / 1327 | 시간 3613[s] | 퍼플렉서티 117.64
| 에폭 5 |  반복 1321 / 1327 | 시간 3669[s] | 퍼플렉서티 115.05
퍼플렉서티 평가 중 ...
209 / 210
-->perplexity:  104.9775086253711
--------------------------------------------------
          :
| 에폭 9 |  반복 1261 / 1327 | 시간 3523[s] | 퍼플렉서티 74.17
| 에폭 9 |  반복 1281 / 1327 | 시간 3579[s] | 퍼플렉서티 73.82
| 에폭 9 |  반복 1301 / 1327 | 시간 3634[s] | 퍼플렉서티 91.67
| 에폭 9 |  반복 1321 / 1327 | 시간 3690[s] | 퍼플렉서티 92.89
퍼플렉서티 평가 중 ...
209 / 210
-->perplexity:  92.12685925148965
--------------------------------------------------
| 에폭 10 |  반복 1241 / 1327 | 시간 3705[s] | 퍼플렉서티 75.02
| 에폭 10 |  반복 1261 / 1327 | 시간 3771[s] | 퍼플렉서티 70.76
| 에폭 10 |  반복 1281 / 1327 | 시간 3835[s] | 퍼플렉서티 71.23
| 에폭 10 |  반복 1301 / 1327 | 시간 3893[s] | 퍼플렉서티 90.29
| 에폭 10 |  반복 1321 / 1327 | 시간 3951[s] | 퍼플렉서티 87.40
퍼플렉서티 평가 중 ...
209 / 210
-->perplexity:  90.84343234152284
--------------------------------------------------
"""
#%%
trainer.plot()

#%%