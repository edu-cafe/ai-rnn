# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:23:03 2020

@author: shkim
"""

"""
## 시계열 데이터 변환을 위한 Toy Example 
* 덧셈(addition) 계산 문제 --> Question & Answering Sentence
* dataset : addition.txt --> seq_dataset.py
"""

#%%
"""
## 시계열 데이터 변환용 덧셈 Toy Dataset 살펴보기
* 덧셈 학습 데이터 : addition.txt --> 5만개의 덧셈 학습 데이터(문제와 답)
"""

from seq_dataset  import load_data, get_vocab

#%%
(x_train, t_train), (x_test, t_test) = load_data('addition.txt', seed=2020)
char_to_id, id_to_char = get_vocab()

print('x_train.shape:', x_train.shape, 't_train.shape:', t_train.shape) # (45000,7),(45000,5)
print('x_test.shape:', x_test.shape, 't_test.shape:', t_test.shape) # (5000,7),(5000,5)

print(x_train[0])  # [ 0  7  2 11 11 12  5]
print(t_train[0])  # [ 6  7  9 10  5]
# print('5(%c)' % id_to_char[5])  # 5( )
# print('6(%c)' % id_to_char[6])  # 6(_)

print('x_train[0]-->', ''.join([id_to_char[c] for c in x_train[0]])) # 19+884 
print('t_train[0]-->', ''.join([id_to_char[c] for c in t_train[0]])) # _903 

#%%
