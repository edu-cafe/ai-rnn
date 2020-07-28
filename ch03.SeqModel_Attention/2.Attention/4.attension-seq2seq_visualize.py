# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:59:16 2020

@author: shkim
"""

"""
# Attension 시각화(Visualization)
"""
#%%
import sys
sys.path.append('..')
import numpy as np
from seq_dataset import load_data, get_vocab
import matplotlib.pyplot as plt
from myutils.attention_seq2seq import AttentionSeq2seq

#%%
(x_train, t_train), (x_test, t_test) = load_data('date.txt')
char_to_id, id_to_char = get_vocab()

# 입력 문장 반전
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

vocab_size = len(char_to_id)  # 59
wordvec_size = 16
hidden_size = 256

model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
model.load_params('AttentionSeq2seq-ep10.pkl')

#%%
_idx = 0
def visualize(attention_map, row_labels, column_labels):
    fig, ax = plt.subplots()
    ax.pcolor(attention_map, cmap=plt.cm.Greys_r, vmin=0.0, vmax=1.0)

    ax.patch.set_facecolor('black')
    ax.set_yticks(np.arange(attention_map.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(attention_map.shape[1])+0.5, minor=False)
    ax.invert_yaxis()
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)

    global _idx
    _idx += 1
    plt.show()

#%%
np.random.seed(1984)
for _ in range(5):
    idx = [np.random.randint(0, len(x_test))]  # len(x_test) : 5000
    x = x_test[idx]
    t = t_test[idx]
    # print('-->x.shape:', x.shape, 't.shape:', t.shape) # (1,29) (1,11)

    model.forward(x, t)
    d = model.decoder.attention.attention_weight  # (N, 1, T) (10, 1, 29)    
    d = np.array(d)  # (10, 1, 29)
    attention_map = d.reshape(d.shape[0], d.shape[2])  # (10, 29)

    # 출력하기 위해 반전
    attention_map = attention_map[:,::-1]
    x = x[:,::-1]

    row_labels = [id_to_char[i] for i in x[0]]  # 29개 문자
    column_labels = [id_to_char[i] for i in t[0]]  # 11개 문자
    column_labels = column_labels[1:]  # y-value, 0:_ 제외한 10개 문자 

    visualize(attention_map, row_labels, column_labels)
