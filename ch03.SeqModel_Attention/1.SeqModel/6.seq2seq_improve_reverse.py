# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:24:24 2020

@author: shkim
"""

"""
## seq2seq 성능 개선
* 1) 입력 데이터 반전(Reverse)
* 2) 엿보기(Peeky)
"""

"""
## seq2seq 성능 개선 : 입력 데이터 반전(Reverse)
* 입력 데이터를 반전시키는 트릭을 사용하면 많은 경우 학습 진행이 빨라져서 
  결과적으로 최종 정확도가 좋아진다고 함 
* "Sequence to sequence learning with neural networks." 
  Advances in neural information processing systems. 2014

* 이전 구현 코드(5.seq2seq_trainer.py)에서 아래 코드의 39번 줄만 추가하면됨 
* 물론, 데이터를 반전시키는 효과는 어떤 문제를 다루느냐에 따라 다르지만, 대부분의 경우 더 좋은 결과로 이어짐

## 왜 입력 데이터를 반전시키는 것만으로 학습의 진행이 빨라지고 정확도가 향상되는 것일까?
* 직관적으로는 기울기 전파가 원활해지기 때문이라고 생각됨 
"""

import numpy as np
import sys
sys.path.append('../../')
from myutils.seq2seq import Seq2seq
from myutils.optimizer import Adam
from myutils.trainer import Trainer
from seq_dataset import load_data, get_vocab

#%%
# read additon dataset
(x_train, t_train), (x_test, t_test) = load_data('addition.txt')
## seq2seq 성능 개선 : 입력 데이터 반전(Reverse)
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]  #*****************

char_to_id, id_to_char = get_vocab()
print(x_train.shape, t_train.shape, x_test.shape, t_test.shape) # (45000,7) (45000,5) (5000,7) (5000,5)
print('vocab_size:', len(id_to_char))  # 13 : 0~9, +, _, ' '

#%%
# Setting hyperparameters
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# Create Model, Optimizer, Trainer
model = Seq2seq(vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

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
# 그래프 그리기
import matplotlib.pyplot as plt

x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('에폭')
plt.ylabel('정확도')
plt.ylim(0, 1.0)
plt.show()

"""
* 결과를 보면 25에폭 학습결과의 정답률은 50% 정도임 --> 입력 데이터를 반전 시킨 것만으로 40%가 개선됨
* 물론, 데이터를 반전시키는 효과는 어떤 문제를 다루느냐에 따라 다르지만, 대부분의 경우 더 좋은 결과로 이어짐
"""
#%%
"""
| 에폭 1 |  반복 1 / 351 | 시간 0[s] | 손실 2.56
| 에폭 1 |  반복 21 / 351 | 시간 0[s] | 손실 2.52
| 에폭 1 |  반복 41 / 351 | 시간 0[s] | 손실 2.17
| 에폭 1 |  반복 61 / 351 | 시간 1[s] | 손실 1.96
| 에폭 1 |  반복 81 / 351 | 시간 1[s] | 손실 1.91
| 에폭 1 |  반복 101 / 351 | 시간 2[s] | 손실 1.87
| 에폭 1 |  반복 121 / 351 | 시간 3[s] | 손실 1.86
| 에폭 1 |  반복 141 / 351 | 시간 3[s] | 손실 1.84
| 에폭 1 |  반복 161 / 351 | 시간 4[s] | 손실 1.80
| 에폭 1 |  반복 181 / 351 | 시간 4[s] | 손실 1.78
| 에폭 1 |  반복 201 / 351 | 시간 5[s] | 손실 1.77
| 에폭 1 |  반복 221 / 351 | 시간 6[s] | 손실 1.77
| 에폭 1 |  반복 241 / 351 | 시간 6[s] | 손실 1.76
| 에폭 1 |  반복 261 / 351 | 시간 7[s] | 손실 1.75
| 에폭 1 |  반복 281 / 351 | 시간 8[s] | 손실 1.74
| 에폭 1 |  반복 301 / 351 | 시간 9[s] | 손실 1.74
| 에폭 1 |  반복 321 / 351 | 시간 9[s] | 손실 1.74
| 에폭 1 |  반복 341 / 351 | 시간 10[s] | 손실 1.73
Q:   58+77 , A: 162  , Predit: 100 
Q: 461+579 , A: 1139 , Predit: 1000
Q:  48+285 , A: 666  , Predit: 1001
Q:   551+8 , A: 163  , Predit: 100 
Q:  55+763 , A: 422  , Predit: 1001
Q: 752+006 , A: 857  , Predit: 1000
Q: 292+167 , A: 1053 , Predit: 1000
Q: 795+038 , A: 1427 , Predit: 1000
Q:  838+62 , A: 864  , Predit: 1001
Q:  39+341 , A: 236  , Predit: 703 
Epoch_1-->Validation Accuracy:0.120 %
             :
| 에폭 25 |  반복 1 / 351 | 시간 0[s] | 손실 0.29
| 에폭 25 |  반복 21 / 351 | 시간 0[s] | 손실 0.29
| 에폭 25 |  반복 41 / 351 | 시간 1[s] | 손실 0.28
| 에폭 25 |  반복 61 / 351 | 시간 1[s] | 손실 0.26
| 에폭 25 |  반복 81 / 351 | 시간 2[s] | 손실 0.26
| 에폭 25 |  반복 101 / 351 | 시간 2[s] | 손실 0.27
| 에폭 25 |  반복 121 / 351 | 시간 3[s] | 손실 0.29
| 에폭 25 |  반복 141 / 351 | 시간 3[s] | 손실 0.28
| 에폭 25 |  반복 161 / 351 | 시간 4[s] | 손실 0.28
| 에폭 25 |  반복 181 / 351 | 시간 4[s] | 손실 0.28
| 에폭 25 |  반복 201 / 351 | 시간 5[s] | 손실 0.27
| 에폭 25 |  반복 221 / 351 | 시간 6[s] | 손실 0.29
| 에폭 25 |  반복 241 / 351 | 시간 6[s] | 손실 0.27
| 에폭 25 |  반복 261 / 351 | 시간 7[s] | 손실 0.28
| 에폭 25 |  반복 281 / 351 | 시간 7[s] | 손실 0.28
| 에폭 25 |  반복 301 / 351 | 시간 8[s] | 손실 0.27
| 에폭 25 |  반복 321 / 351 | 시간 8[s] | 손실 0.28
| 에폭 25 |  반복 341 / 351 | 시간 9[s] | 손실 0.28
Q:   58+77 , A: 162  , Predit: 162 
Q: 461+579 , A: 1139 , Predit: 1140
Q:  48+285 , A: 666  , Predit: 666 
Q:   551+8 , A: 163  , Predit: 163 
Q:  55+763 , A: 422  , Predit: 422 
Q: 752+006 , A: 857  , Predit: 856 
Q: 292+167 , A: 1053 , Predit: 1052
Q: 795+038 , A: 1427 , Predit: 1426
Q:  838+62 , A: 864  , Predit: 864 
Q:  39+341 , A: 236  , Predit: 236 
Epoch_25-->Validation Accuracy:54.320 %
"""




