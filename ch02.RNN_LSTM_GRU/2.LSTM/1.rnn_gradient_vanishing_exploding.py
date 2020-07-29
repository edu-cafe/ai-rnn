# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:57:14 2020

@author: shkim
"""

"""
## 기본 RNN의 문제점 
* 기울기 폭발 (Exploding Gradients) --> 그래프 그리기
* 기울기 소실 (Vanishing Gradients) --> 그래프 그리기 
"""

import numpy as np

#%%
N = 2    # mini-batch size
H = 3    # hidden state vector dimenson
T = 20   # Time-series data length

dh = np.ones((N, H))
np.random.seed(3)
# Wh = np.random.randn(H, H)  # Exploding Gradiets
Wh = np.random.randn(H, H) * 0.5   # Vanishing Gradiets

norm_list = []
for t in range(T):
    dh = np.matmul(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)
    
#%%
# 그래프 그리기
import matplotlib.pyplot as plt
plt.rc('font', family ='Malgun Gothic')
    
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
plt.xlabel('시간 크기(time step)')
plt.ylabel('노름(norm)')
plt.show()

#%%