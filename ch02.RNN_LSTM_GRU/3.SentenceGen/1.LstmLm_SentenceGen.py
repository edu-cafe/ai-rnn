# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 10:52:28 2020

@author: shkim
"""

"""
# 언어 모델을 사용한 문장 생성
## LSTM(RNN)을 사용한 문장 생성의 순서
* 1) 특정 단어를 입력으로 넣음
* 2) 입력 단어 다음에 올 두 번째 단어를 학습된 언어 모델을 이용하여 축출(샘플링)함
* 3) 위에서 축출(샘플링)한 두번째 단어를 다시 언어 모델에 입력하고 또 다음 단어를 얻음
* 4) 이러한 작업을 원하는 만큼 반복하거나 <eos>와 같은 종결기호가 나타날 때까지 반복함

## 학습된 언어 모델로부터 단어를 선택하는 방법
* 1) 결정적(deterministic) 방법 : 선택되는 단어가 확률이 가장 높은 단어로 정해짐
* 2) 확률적(probabilistic) 방법 : 선택되는 단어는 실행 시마다 달라질 수 있음 
"""
#%%
"""
## LstmLm 언어 모델을 이용한 문장 생성 클래스 구현
* myutils.time_layers --> LstmLmGen
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.time_layers import LstmLm
from myutils.functions import softmax

#%%
class LstmLmGen(LstmLm):
    # skip_ids : <unk>, N등 전처리된 데이터를 샘플링하지 않게 하는 용도
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]
        
        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1,1)  # mini-batch 처리에 맞게 reshape
            score = self.predict(x)
            p = softmax(score.flatten())
            
            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
                
        return word_ids

#%%
"""
## LstmLmGen 클래스를 이용한 문장 생성 구현
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.time_layers import LstmLmGen
from ptb_dataset import load_data

#%%
corpus, word_to_id, id_to_word = load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = LstmLmGen()
# model.load_params('./LstmLm-ep4.pkl')

# Setting start_id and skip_ids
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]

# Generate Sentence
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)



#%%








