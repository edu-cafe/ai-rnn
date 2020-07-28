# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:06:38 2020

@author: shkim
"""

"""
# AttentionSeq2seq 응용 : 날짜 형식 변환 문제
## 문제 목표
* '날짜 형식'을 변경하는 문제(데이터 크기가 작고, 어느 쪽인가를 맞추는 인위적인 문제)
* 영어권에서 사용되는 다양한 날짜 형식을 표준 형식으로 변환하는 것이 목표임
  - 'september 27, 1994' --> '1994-09-27'
  - 'JUN 17, 2013' --> '2013-06-17'
  - '2/10/93' --> '1993-02-10'

## 데이터 셋
* date.txt
* 날짜 변환 학습데이터를 50,000개 담고 있음
  - x:'september 27, 1994bbbbbbbbbbb'   y: '_1994-09-27'
  - x:'2/10/93bbbbbbbbbbbbbbbbbbbbbb'   y: '_1993-02-10'

"""

"""
## 번역 용 데이터셋
* 번역 용 데이터셋 중에서는 'WMT'가 유명함
* 이 데이터셋에는 영어와 프랑스어(또는 영어와 독일어) 학습 데이터가 쌍으로 준비되어 있음
* WMT 데이터셋은 많은 연구에서 벤치마크로 이용되고 있으며, seq2seq 성능 평가로도 자주 이용됨
* 다만, WMT 데이터셋은 덩치가 커서(20GB 이상) 부담이 됨 
"""

#%%
"""
# Attention seq2seq의 학습 코드 구현
*
"""
import numpy as np
import sys
sys.path.append('../../')
from seq_dataset import load_data, get_vocab
from myutils.optimizer import Adam
from myutils.trainer import Trainer
from myutils.attention_seq2seq import AttentionSeq2seq

#%%
def eval_correct(model, question, correct, id_to_char, verbose=False):
    correct = correct.flatten()
    
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))
    
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])
    if verbose: print('Q:', question, ', A:', correct, ', Predit:', guess)
    
    return 1 if guess == correct else 0

#%%
# read additon dataset
(x_train, t_train), (x_test, t_test) = load_data('date.txt')

# Data Reverse
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]  

char_to_id, id_to_char = get_vocab()
print(x_train.shape, t_train.shape, x_test.shape, t_test.shape) # (45000, 29) (45000, 11) (5000, 29) (5000, 11)
print('vocab_size:', len(id_to_char))  # 59

#%%
# Setting hyperparameters
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

# Create Model, Optimizer, Trainer
model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

#%%
acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    
    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_correct(model, question, correct, id_to_char, verbose)
    
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('Epoch_%d-->Validation Accuracy:%.3f %%' % (epoch+1, acc * 100))

#%%   
model.save_params()

#%%
# 그래프 그리기
import matplotlib.pyplot as plt

x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('에폭')
plt.ylabel('정확도')
plt.ylim(0, 1.1)
plt.show()

#%%
"""
* 실행 결과를 보면 세 번째 epoch에는 거의 모든 문제를 풀어내는 아주 좋은 결과를 보여줌 
* 단순한 base seq2seq 모델, peeky seq2seq 모델, attention seq2seq를 비교해 보면,
  base 모델은 전혀 쓸모가 없음을 알 수 있고,
  peeky도 좋은 결과를 보여주지만 학습 속도 측면에서 attention 모델이 우수함
"""

#%%
"""
| 에폭 1 |  반복 1 / 351 | 시간 0[s] | 손실 4.08
| 에폭 1 |  반복 21 / 351 | 시간 13[s] | 손실 3.09
| 에폭 1 |  반복 41 / 351 | 시간 28[s] | 손실 1.90
| 에폭 1 |  반복 61 / 351 | 시간 42[s] | 손실 1.72
           :
| 에폭 1 |  반복 301 / 351 | 시간 214[s] | 손실 1.00
| 에폭 1 |  반복 321 / 351 | 시간 228[s] | 손실 1.00
| 에폭 1 |  반복 341 / 351 | 시간 242[s] | 손실 1.00
Q:                      49/51/01 , A: 1994-10-15 , Predit: 1978-08-11
Q:   8002 ,31 rebmevon ,yadsruht , A: 2008-11-13 , Predit: 1978-08-11
Q:                  3002 ,52 raM , A: 2003-03-25 , Predit: 1978-08-11
Q:    6102 ,22 rebmevoN ,yadseuT , A: 2016-11-22 , Predit: 1978-08-11
Q:       0791 ,81 yluJ ,yadrutaS , A: 1970-07-18 , Predit: 1978-08-11
Q:               2991 ,6 rebotco , A: 1992-10-06 , Predit: 1978-08-11
Q:                       80/32/8 , A: 2008-08-23 , Predit: 1978-08-11
Q:                       70/03/8 , A: 2007-08-30 , Predit: 1978-08-11
Q:                      31/82/01 , A: 2013-10-28 , Predit: 1978-08-11
Q:      6102 ,6 rebmevon ,yadnus , A: 2016-11-06 , Predit: 1978-08-11
Epoch_1-->Validation Accuracy:0.000 %

| 에폭 2 |  반복 301 / 351 | 시간 222[s] | 손실 0.66
| 에폭 2 |  반복 321 / 351 | 시간 236[s] | 손실 0.58
| 에폭 2 |  반복 341 / 351 | 시간 254[s] | 손실 0.46
Q:                      49/51/01 , A: 1994-10-15 , Predit: 1994-10-15
Q:   8002 ,31 rebmevon ,yadsruht , A: 2008-11-13 , Predit: 2006-11-13
Q:                  3002 ,52 raM , A: 2003-03-25 , Predit: 2003-03-25
Q:    6102 ,22 rebmevoN ,yadseuT , A: 2016-11-22 , Predit: 2016-11-22
Q:       0791 ,81 yluJ ,yadrutaS , A: 1970-07-18 , Predit: 1970-07-18
Q:               2991 ,6 rebotco , A: 1992-10-06 , Predit: 1992-10-06
Q:                       80/32/8 , A: 2008-08-23 , Predit: 2008-08-23
Q:                       70/03/8 , A: 2007-08-30 , Predit: 2007-08-09
Q:                      31/82/01 , A: 2013-10-28 , Predit: 1983-10-28
Q:      6102 ,6 rebmevon ,yadnus , A: 2016-11-06 , Predit: 2016-11-08
Epoch_2-->Validation Accuracy:51.640 %

| 에폭 3 |  반복 321 / 351 | 시간 236[s] | 손실 0.01
| 에폭 3 |  반복 341 / 351 | 시간 250[s] | 손실 0.01
Q:                      49/51/01 , A: 1994-10-15 , Predit: 1994-10-15
Q:   8002 ,31 rebmevon ,yadsruht , A: 2008-11-13 , Predit: 2008-11-13
Q:                  3002 ,52 raM , A: 2003-03-25 , Predit: 2003-03-25
Q:    6102 ,22 rebmevoN ,yadseuT , A: 2016-11-22 , Predit: 2016-11-22
Q:       0791 ,81 yluJ ,yadrutaS , A: 1970-07-18 , Predit: 1970-07-18
Q:               2991 ,6 rebotco , A: 1992-10-06 , Predit: 1992-10-06
Q:                       80/32/8 , A: 2008-08-23 , Predit: 2008-08-23
Q:                       70/03/8 , A: 2007-08-30 , Predit: 2007-08-30
Q:                      31/82/01 , A: 2013-10-28 , Predit: 2013-10-28
Q:      6102 ,6 rebmevon ,yadnus , A: 2016-11-06 , Predit: 2016-11-06
Epoch_3-->Validation Accuracy:99.900 %

| 에폭 10 |  반복 301 / 351 | 시간 181[s] | 손실 0.00
| 에폭 10 |  반복 321 / 351 | 시간 194[s] | 손실 0.00
| 에폭 10 |  반복 341 / 351 | 시간 206[s] | 손실 0.00
Q:                      49/51/01 , A: 1994-10-15 , Predit: 1994-10-15
Q:   8002 ,31 rebmevon ,yadsruht , A: 2008-11-13 , Predit: 2008-11-13
Q:                  3002 ,52 raM , A: 2003-03-25 , Predit: 2003-03-25
Q:    6102 ,22 rebmevoN ,yadseuT , A: 2016-11-22 , Predit: 2016-11-22
Q:       0791 ,81 yluJ ,yadrutaS , A: 1970-07-18 , Predit: 1970-07-18
Q:               2991 ,6 rebotco , A: 1992-10-06 , Predit: 1992-10-06
Q:                       80/32/8 , A: 2008-08-23 , Predit: 2008-08-23
Q:                       70/03/8 , A: 2007-08-30 , Predit: 2007-08-30
Q:                      31/82/01 , A: 2013-10-28 , Predit: 2013-10-28
Q:      6102 ,6 rebmevon ,yadnus , A: 2016-11-06 , Predit: 2016-11-06
Epoch_10-->Validation Accuracy:99.960 %
"""







