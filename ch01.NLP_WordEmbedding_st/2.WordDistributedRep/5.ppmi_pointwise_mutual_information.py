# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:10:09 2020

@author: shkim
"""

import numpy as np

#%%
# myutils.util
## PPMI(Positive Pointwise Mutual Information) 함수 구현 
# C:동시발생 행렬
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j]*N / (S[j]*S[i]) + eps)
            M[i,j] = max(0, pmi)
            
            if verbose:
                cnt += 1
                if cnt % (total//3) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
    
    return M

#%%
import sys
sys.path.append('../../')
from myutils.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

print(C.shape)

#%%
W = ppmi(C)

np.set_printoptions(precision=3)
print(word_to_id.keys())
print(W)

#%%












