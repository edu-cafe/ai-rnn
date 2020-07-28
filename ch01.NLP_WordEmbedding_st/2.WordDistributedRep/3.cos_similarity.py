# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:42:56 2020

@author: shkim
"""

import numpy as np
import sys
sys.path.append('../../')
from myutils.util import preprocess, create_co_matrix

#%%
# myutils.util
def cos_similarity(x, y):
    nx = x / np.sqrt(np.sum(x**2))  # x의 정규화 
    ny = y / np.sqrt(np.sum(y**2))  # y의 정규화 
    return np.dot(nx, ny)

#%%
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(C)

#%%
c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))  # 0.7071067811865475

#%%






