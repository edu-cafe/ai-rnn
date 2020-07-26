# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:58:53 2020

@author: shkim
"""

#%%
"""
## 학습 데이터의 준비
* 입력 : word2vec에서 이용하는 신경망의 입력은 **'맥락'**임
* 타깃 : word2vec의 정답 레이블은 **맥락에 둘러사인 중앙의 단어, 즉 '타깃'**임
* 우리가 해야할 일은 신경망에 '맥락'을 입력했을 때 '타깃'이 출현할 확률을 높이는 것임 
"""

#%%
import sys
sys.path.append('../../')
from myutils.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)  
print(word_to_id)

#%%
a = [1,2,3,4,5,6,7]
print(a[1:-1])  # [2, 3, 4, 5, 6]
print(a[2:-1])  # [3, 4, 5, 6]
print(a[2:-2])  # [3, 4, 5]

#%%
"""
### 학습데이터의 준비
* 말뭉치(corpus)로부터 맥락과 타깃 축출 
"""
# myutils.util
import numpy as np

def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    # print('-->target:', target)
    contexts = []
    
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
        # print('-->contexts:', contexts)
        
    return np.array(contexts), np.array(target)
            
#%%
import sys
sys.path.append('../../')
from myutils.util import preprocess
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)  # [0, 1, 2, 3, 4, 1, 5, 6]

contexts, target = create_contexts_target(corpus, window_size=1)

print(contexts)  # [[0 2] [1 3] [2 4] [3 1] [4 5] [1 6]]
print(target)  # [1 2 3 4 1 5]

#%%
"""
### 학습데이터의 준비
* 맥락과 타깃을 단어 ID로부터 One-Hot 표현으로 변환 
* 맥락의 shape : (6,2) --> (6,2,7)
"""
# myutils.util
def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환
    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot

#%%
import sys
sys.path.append('../../')
from myutils.util import preprocess
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)
# print(target)  # [1 2 3 4 1 5]
# print(contexts)  # [[0 2] [1 3] [2 4] [3 1] [4 5] [1 6]]

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

print(target)
print(contexts)

#%%
"""
## CBOW 모델 구현
* # myutils.layers
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        
        # Weight Initialize
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # Layers Create
        self.in_layer0 = ......
        self.in_layer1 = ......
        self.out_layer = ......
        self.loss_layer = ..........
        
        # Aggregate all Parameters and Gradients
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # Save word-vector(단어의 분산 표현)
        self.word_vecs = .....
        
    
    # contexts : 3-dim ndarray, (6, 2, 7)
    # (6, 2, 7) : 6(mini-batch size), 2(context window size), 7(one-hot vector)
    # target : 2-dim ndarray, (6, 7)
    def forward(self, contexts, target):
        h0 = self.in_layer0.__________
        h1 = self.in_layer1.__________
        h = (h0 + h1) * ____
        score = self.out_layer._____
        loss = self.loss_layer._______
        return loss
    
    
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

#%%
"""
## CBOW 모델 학습(training) 코드 구현 
*
"""

import sys
sys.path.append('../../')
from myutils.util import preprocess, create_contexts_target, convert_one_hot
# from myutils.layers import SimpleCBOW
from myutils.optimizer import Adam
from myutils.trainer import Trainer

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = ________(text)
vocab_size = len(word_to_id)

contexts, target = _____________(corpus, window_size)
print(target)  # [1 2 3 4 1 5]
print(contexts)  # [[0 2] [1 3] [2 4] [3 1] [4 5] [1 6]]

#%%
target = ________(target, vocab_size)
contexts = ________(contexts, vocab_size)
print('target_shape:', target.shape)  # (6, 7)
print('contexts_shape:', contexts.shape)  # (6, 2, 7)

#%%
model = .........
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.____(contexts, target, max_epoch, batch_size)

#%%
trainer.plot()

#%%
"""
## 학습이 끝난 후의 가중치 매개변수 출력 
* 입력 측 MatMul 계층의 가중치는 instance 변수 word_vecs에 저장되어 있음 
* word_vecs의 각 행에는 대응하는 단어 ID의 분산 표현이 저장되어 있음 
* 단어를 밀집벡터(분산 표현)로 나타낼 수 있음 : 단어의 의미를 잘 파악한 벡터 표현 
"""

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():    
    print(word, word_vecs[word_id])

"""
you [ 1.0091473  0.9849929 -1.6978021 -1.0196477 -0.9925592]
say [-1.215253   -1.263818   -1.0873705   0.03718124  0.24266285]
goodbye [ 0.8949067   0.9265815   0.80065316 -0.9914794  -0.9931721 ]
and [-1.0043075  -1.0876608  -0.83345413 -1.5787815   1.5873108 ]
i [ 0.87272155  0.9171881   0.81294656 -0.9443994  -0.9713344 ]
hello [ 1.024529   0.9908122 -1.7075027 -1.0413941 -0.9901753]
. [-1.0638249 -1.0150329 -1.0572715  1.3554877 -1.4923285]
"""
#%%



