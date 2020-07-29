# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 09:58:05 2020

@author: shkim
"""

"""
## 개선된 seq2seq 모델을 사용한 평가
* 입력 문장으 반전시키는 Reverse 적용
* Encoder의 정보를 널리 퍼지게 하는 Peeky 적용
* 실험결과 98%이상의 정확도가 달성됨을 확인할 수 있음
* 그러나, Peeky를 이용하게되면 우리의 신경망은 가중치 매개변수가 커져서 계산량도 늘어남

### seq2seq의 학습은 기본적인 신경망의 학습과 같은 흐름으로 이뤄짐 
* 1) 학습 데이터에서 미니배치를 선택하고,
* 2) 미니배치로부터 기울기를 계산하고,
* 3) 기울기를 사용하여 매개변수를 갱신한다.
"""

import numpy as np
import sys
sys.path.append('..')
from myutils.seq2seq import Encoder
from myutils.seq2seq_peeky import Seq2seqPeeky
from myutils.optimizer import Adam
from myutils.trainer import Trainer
from seq_dataset import load_data, get_vocab

#%%
# read additon dataset
(x_train, t_train), (x_test, t_test) = load_data('addition.txt')
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]  # Data Reverse

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
model = Seq2seqPeeky(vocab_size, wordvec_size, hidden_size)
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
* 결과를 보면 정답률은 98%로 100%에 가까워졌음
* 
"""
#%%
"""
| 에폭 25 |  반복 1 / 351 | 시간 0[s] | 손실 0.02
| 에폭 25 |  반복 21 / 351 | 시간 0[s] | 손실 0.02
| 에폭 25 |  반복 41 / 351 | 시간 1[s] | 손실 0.01
| 에폭 25 |  반복 61 / 351 | 시간 1[s] | 손실 0.01
| 에폭 25 |  반복 81 / 351 | 시간 2[s] | 손실 0.02
| 에폭 25 |  반복 101 / 351 | 시간 3[s] | 손실 0.02
| 에폭 25 |  반복 121 / 351 | 시간 3[s] | 손실 0.02
| 에폭 25 |  반복 141 / 351 | 시간 4[s] | 손실 0.02
| 에폭 25 |  반복 161 / 351 | 시간 5[s] | 손실 0.02
| 에폭 25 |  반복 181 / 351 | 시간 5[s] | 손실 0.02
| 에폭 25 |  반복 201 / 351 | 시간 6[s] | 손실 0.02
| 에폭 25 |  반복 221 / 351 | 시간 6[s] | 손실 0.02
| 에폭 25 |  반복 241 / 351 | 시간 7[s] | 손실 0.02
| 에폭 25 |  반복 261 / 351 | 시간 8[s] | 손실 0.02
| 에폭 25 |  반복 281 / 351 | 시간 8[s] | 손실 0.02
| 에폭 25 |  반복 301 / 351 | 시간 9[s] | 손실 0.03
| 에폭 25 |  반복 321 / 351 | 시간 10[s] | 손실 0.03
| 에폭 25 |  반복 341 / 351 | 시간 10[s] | 손실 0.02
Q:   58+77 , A: 162  , Predit: 162 
Q: 461+579 , A: 1139 , Predit: 1139
Q:  48+285 , A: 666  , Predit: 666 
Q:   551+8 , A: 163  , Predit: 163 
Q:  55+763 , A: 422  , Predit: 422 
Q: 752+006 , A: 857  , Predit: 857 
Q: 292+167 , A: 1053 , Predit: 1053
Q: 795+038 , A: 1427 , Predit: 1427
Q:  838+62 , A: 864  , Predit: 864 
Q:  39+341 , A: 236  , Predit: 236 
Epoch_25-->Validation Accuracy:98.020 %
"""




