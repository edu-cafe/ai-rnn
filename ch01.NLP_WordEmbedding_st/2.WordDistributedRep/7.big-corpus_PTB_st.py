# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:58:34 2020

@author: shkim
"""

"""
## PTB(Penn TreeBank) : 비교적 큰 말뭉치(corpus)
  https://github.com/tomsercu/lstm/tree/master/data
* PTB 말뭉치는 텍스트 파일로 제공되며, 원래의 PTB 문장에 몇가지 전처리를 해 두었음
  희소한 단어를 <unk>라는 특수 문자로 치환한다거나 구체적인 숫자를 'N'으로 대체 등
"""

#%%
from ptb_dataset import load_data

corpus, word_to_id, id_to_word = load_data('train')

#%%
print('말뭉치 크기:', len(corpus), 'len(word_to_id):', 
      len(word_to_id), 'len(id_to_word):',len(id_to_word))
print('corpus[:30]:', corpus[:30])
print('-'*50)
print('id_to_word[0]:', id_to_word[0])
print('id_to_word[20]:', id_to_word[20])
print('id_to_word[100]:', id_to_word[100])
print('-'*50)
print("word_to_id['car']:", word_to_id['car'])
print("word_to_id['happy']:", word_to_id['happy'])
print("word_to_id['lexus']:", word_to_id['lexus'])

#%%
"""
## PTB 데이터셋 평가
* PTB 데이터셋에 통계 기반 기법을 적용해 보자
* 큰 행렬에 SVD를 적용해야 하므로 고속 SVD를 이용할 것임 : sklearn  
"""
import numpy as np
from ptb_dataset import load_data
import sys
sys.path.append('../../')
from myutils.util import create_co_matrix, ppmi, most_similar

#%%
corpus, word_to_id, id_to_word = load_data('train')
vocab_size = ......
print('vocab_size:', vocab_size)  # 10000

#%%
print('동시 발생 수 계산...')
window_size = 2
C = _________(corpus, vocab_size, window_size)
print(C.shape)  # (10000, 10000)

#%%
print('PPMI 계산...')
W = _____(C, verbose=True)
print(W.shape)  # (10000, 10000)

#%%
wordvec_size = 100
try:
    # truncated SVD
    print('Truncated SVD 계산...')
    from sklearn.utils.extmath import randomized_svd
    U, S, V = _______(__, n_components=wordvec_size, 
                             n_iter=5, random_state=None)
except ImportError:
    # SVD
    print('SVD 계산...')
    U, S, V = np.linalg.svd(W)
    
word_vecs = ......
print(word_vecs.shape)  # (10000, 100)

#%%
print('Query Similarity...')
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    ________(query, word_to_id, id_to_word, _______, top=5)


#%%
    











