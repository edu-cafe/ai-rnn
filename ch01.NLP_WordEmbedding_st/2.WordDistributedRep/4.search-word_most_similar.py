# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:05:32 2020

@author: shkim
"""

"""
## 유사 단어의 랭킹 표시
* 어떤 단어가 검색어로 주어지면, 그 검색어와 비슷한 단어를 유사도 순으로 출력하는 함수 구현 
* most_similar(query, word_to_id, id_to_word, word_matrix, top=5)
"""

import numpy as np
import sys
sys.path.append('../../')
from myutils.util import cos_similarity

#%%
# myutils.util
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 1. 검색어를 꺼냄
    if query not in word_to_id:
        print('%s(을)를 찾을 수 없습니다.', query)
        return
    
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    # 2. 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    # 3. 코사인 유사도를 기준으로 내림차순으로 출력 
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print('%s: %s' % (id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            return

#%%
x = np.array([100, -20, 1])
print(x.argsort())
print((-x).argsort())

#%%
import sys
sys.path.append('../../')
from myutils.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)

#%%





