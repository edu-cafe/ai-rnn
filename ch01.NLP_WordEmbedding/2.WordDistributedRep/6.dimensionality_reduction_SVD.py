# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:28:53 2020

@author: shkim
"""

"""
## 차원 감소(Dimensionality Reduction)
## SVD(Singular Value Decomposition, 특잇값분해)
* np.linalg.svd() 사용 
"""

#%%
import numpy as np
import sys
sys.path.append('../../')
from myutils.util import preprocess, create_co_matrix, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C, verbose=False)

U, S, V = np.linalg.svd(W)  # (7, 7) (7,) (7, 7)

print(U.shape, S.shape, V.shape)

np.set_printoptions(precision=3)
print(C[0])  # [0 1 0 0 0 0 0]
print(W[0])  # [0.        1.8073549 0.        0.        0.        0.        0.       ]
print(U[0])  # [-3.4094876e-01 -1.1102230e-16 -3.8857806e-16 -1.2051624e-01  0.0000000e+00  9.3232495e-01  2.2259700e-16]
# 희소벡터 W가 밀집벡터 U로 변경됨

print(U[0, :2])  # [-3.4094876e-01 -1.1102230e-16]
# 밀집벡터의 차원을 감소시키려면, 예를들어 2차원 벡터로 줄이려면 단순히 처음 두 원소를 꺼내면 됨

#%%
print(S)
print(U[:])
print(V[:])

#%%
"""
* 각 단어를 2차원벡터로 표현한 후 그래프로 그려보자.
"""
import matplotlib.pyplot as plt

plt.scatter(U[:,0], U[:,1], alpha=0.5)

for word, word_id in word_to_id.items():
    print(word)
    plt.annotate(word, (U[word_id, 0], U[word_id, 1])) 

plt.show()

#%%
"""
* 각 단어를 2차원벡터로 표현한 후 그래프로 그려보자.
"""
import matplotlib.pyplot as plt

for word, word_id in word_to_id.items():
    plt.scatter(U[:,0], U[:,1], alpha=0.5)
    print(word)
    plt.annotate(word, (U[word_id, 0], U[word_id, 1])) 
    plt.show()
    input('Enter to continue..')

#%%


