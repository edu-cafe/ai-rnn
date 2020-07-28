# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 09:58:05 2020

@author: shkim
"""

"""
## seq2seq 평가
* 매 epoch마다 seq2seq가 테스트 데이터를 풀게 하여(문자열 생성을 수행하여) 학습 중간 중간 정답률을 측정할 것임
### seq2seq의 학습은 기본적인 신경망의 학습과 같은 흐름으로 이뤄짐 
* 1) 학습 데이터에서 미니배치를 선택하고,
* 2) 미니배치로부터 기울기를 계산하고,
* 3) 기울기를 사용하여 매개변수를 갱신한다.
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
* 결과를 보면 25에폭 학습결과의 정답률은 10% 정도임 --> 개선이 필요함
"""
#%%
"""
| 에폭 1 |  반복 1 / 351 | 시간 0[s] | 손실 2.56
| 에폭 1 |  반복 21 / 351 | 시간 0[s] | 손실 2.53
| 에폭 1 |  반복 41 / 351 | 시간 1[s] | 손실 2.17
| 에폭 1 |  반복 61 / 351 | 시간 1[s] | 손실 1.96
| 에폭 1 |  반복 81 / 351 | 시간 2[s] | 손실 1.92
| 에폭 1 |  반복 101 / 351 | 시간 2[s] | 손실 1.87
| 에폭 1 |  반복 121 / 351 | 시간 3[s] | 손실 1.85
| 에폭 1 |  반복 141 / 351 | 시간 3[s] | 손실 1.83
| 에폭 1 |  반복 161 / 351 | 시간 4[s] | 손실 1.79
| 에폭 1 |  반복 181 / 351 | 시간 4[s] | 손실 1.77
| 에폭 1 |  반복 201 / 351 | 시간 5[s] | 손실 1.77
| 에폭 1 |  반복 221 / 351 | 시간 6[s] | 손실 1.76
| 에폭 1 |  반복 241 / 351 | 시간 6[s] | 손실 1.76
| 에폭 1 |  반복 261 / 351 | 시간 7[s] | 손실 1.76
| 에폭 1 |  반복 281 / 351 | 시간 7[s] | 손실 1.75
| 에폭 1 |  반복 301 / 351 | 시간 8[s] | 손실 1.74
| 에폭 1 |  반복 321 / 351 | 시간 8[s] | 손실 1.75
| 에폭 1 |  반복 341 / 351 | 시간 9[s] | 손실 1.74
Q: 77+85   , A: 162  , Predit: 100 
Q: 975+164 , A: 1139 , Predit: 1000
Q: 582+84  , A: 666  , Predit: 1000
Q: 8+155   , A: 163  , Predit: 100 
Q: 367+55  , A: 422  , Predit: 1000
Q: 600+257 , A: 857  , Predit: 1000
Q: 761+292 , A: 1053 , Predit: 1000
Q: 830+597 , A: 1427 , Predit: 1000
Q: 26+838  , A: 864  , Predit: 1000
Q: 143+93  , A: 236  , Predit: 100 
Epoch_1-->Validation Accuracy:0.180 %
             :
| 에폭 25 |  반복 1 / 351 | 시간 0[s] | 손실 0.73
| 에폭 25 |  반복 21 / 351 | 시간 0[s] | 손실 0.77
| 에폭 25 |  반복 41 / 351 | 시간 1[s] | 손실 0.79
| 에폭 25 |  반복 61 / 351 | 시간 1[s] | 손실 0.80
| 에폭 25 |  반복 81 / 351 | 시간 2[s] | 손실 0.79
| 에폭 25 |  반복 101 / 351 | 시간 3[s] | 손실 0.76
| 에폭 25 |  반복 121 / 351 | 시간 3[s] | 손실 0.78
| 에폭 25 |  반복 141 / 351 | 시간 4[s] | 손실 0.78
| 에폭 25 |  반복 161 / 351 | 시간 5[s] | 손실 0.75
| 에폭 25 |  반복 181 / 351 | 시간 5[s] | 손실 0.76
| 에폭 25 |  반복 201 / 351 | 시간 6[s] | 손실 0.81
| 에폭 25 |  반복 221 / 351 | 시간 7[s] | 손실 0.77
| 에폭 25 |  반복 241 / 351 | 시간 7[s] | 손실 0.78
| 에폭 25 |  반복 261 / 351 | 시간 8[s] | 손실 0.76
| 에폭 25 |  반복 281 / 351 | 시간 9[s] | 손실 0.77
| 에폭 25 |  반복 301 / 351 | 시간 9[s] | 손실 0.76
| 에폭 25 |  반복 321 / 351 | 시간 10[s] | 손실 0.75
| 에폭 25 |  반복 341 / 351 | 시간 10[s] | 손실 0.76
Q: 77+85   , A: 162  , Predit: 162 
Q: 975+164 , A: 1139 , Predit: 1139
Q: 582+84  , A: 666  , Predit: 662 
Q: 8+155   , A: 163  , Predit: 163 
Q: 367+55  , A: 422  , Predit: 419 
Q: 600+257 , A: 857  , Predit: 856 
Q: 761+292 , A: 1053 , Predit: 1059
Q: 830+597 , A: 1427 , Predit: 1441
Q: 26+838  , A: 864  , Predit: 858 
Q: 143+93  , A: 236  , Predit: 239 
Epoch_25-->Validation Accuracy:10.840 %
"""




