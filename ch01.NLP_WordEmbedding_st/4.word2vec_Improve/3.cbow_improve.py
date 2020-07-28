# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:38:30 2020

@author: shkim
"""

"""
# word2vec 성능 개선 
## 개선된 CBOW 모델 구현
* Embedding Layer와 NegativeSamplingLoss Layer 적용함 
* 맥락의 윈도우 크기를 임의로 조절할 수 있도록 확장함 
"""

#%%
import numpy as np
import sys
sys.path.append('../../')
from myutils.layers import Embedding
from myutils.negative_sampling_layer import NegativeSamplingLoss

class CBOW:
    # 맥락 단어의 크기 : window_size * 2
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        
        # Weight Initialize
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')
        
        # Layer Create
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)
        
        # Aggregate all Weights and Gradients
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # Save word distributed representation (word vectors)
        self.word_vecs = W_in
    
    # contexts : 맥락 단어 ID, 2차원 배열 
    # target : 타깃 단어 ID, 1차원 배열 
    def forward(self, context, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(context[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

#%%
"""
## CBOW 모델 Training 코드 구현
"""
import numpy as np
import pickle
import sys
sys.path.append('../../')
from myutils.trainer import Trainer
from myutils.optimizer import Adam
from myutils.cbow import CBOW
from myutils.util import create_contexts_target
from ptb_dataset import load_data

#%%
# hyper-parameter setting
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 20

# Data Read
corpus, word_to_id, id_to_word = load_data('train')
vocab_size = len(word_to_id)
print(corpus.shape, vocab_size)  # (929589,)   10000

#%%
# Create contexts and target
contexts, target = create_contexts_target(corpus, window_size)
print(contexts.shape, target.shape)  # (929579, 10) (929579,)

#%%
# Create Model
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer) 

#%%
# Start Training
trainer.fit(contexts, target, max_epoch, batch_size)

"""
| 에폭 1 |  반복 1 / 9295 | 시간 0[s] | 손실 4.16
| 에폭 1 |  반복 21 / 9295 | 시간 1[s] | 손실 4.16
| 에폭 1 |  반복 41 / 9295 | 시간 3[s] | 손실 4.15
| 에폭 1 |  반복 61 / 9295 | 시간 4[s] | 손실 4.12
     :
| 에폭 20 |  반복 9141 / 9295 | 시간 13794[s] | 손실 1.16
| 에폭 20 |  반복 9161 / 9295 | 시간 13795[s] | 손실 1.20
| 에폭 20 |  반복 9181 / 9295 | 시간 13797[s] | 손실 1.18
| 에폭 20 |  반복 9201 / 9295 | 시간 13798[s] | 손실 1.14
| 에폭 20 |  반복 9221 / 9295 | 시간 13800[s] | 손실 1.18
| 에폭 20 |  반복 9241 / 9295 | 시간 13801[s] | 손실 1.19
| 에폭 20 |  반복 9261 / 9295 | 시간 13802[s] | 손실 1.18
| 에폭 20 |  반복 9281 / 9295 | 시간 13804[s] | 손실 1.23
"""

#%%
trainer.plot()

#%%
# 나중에 사용할 수 있도록 필요한 데이터 저장
word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)


#%%
"""
## CBOW 모델 평가
"""
import pickle
import sys
sys.path.append('../../')
from myutils.util import most_similar

pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']
    
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    
"""
[query] you
we: 0.6796875
i: 0.64404296875
they: 0.5791015625
your: 0.525390625
weird: 0.5107421875

[query] year
month: 0.77197265625
week: 0.75048828125
summer: 0.66796875
spring: 0.65869140625
decade: 0.6455078125

[query] car
auto: 0.55126953125
cars: 0.49462890625
luxury: 0.474609375
trucks: 0.4736328125
chevrolet: 0.44189453125

[query] toyota
engines: 0.55419921875
trucks: 0.5224609375
fuel: 0.51513671875
beretta: 0.51318359375
minivans: 0.5068359375
"""

#%%
import sys
sys.path.append('../../')
from myutils.util import analogy

analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs)

analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)

analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)

analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)

"""
[analogy] king:man = queen:?
 woman: 5.16015625
 veto: 4.9296875
 ounce: 4.69140625
 earthquake: 4.6328125
 successor: 4.609375

[analogy] take:took = go:?
 went: 4.55078125
 points: 4.25
 began: 4.09375
 comes: 3.98046875
 oct.: 3.90625

[analogy] car:cars = child:?
 children: 5.21875
 average: 4.7265625
 yield: 4.20703125
 cattle: 4.1875
 priced: 4.1796875

[analogy] good:better = bad:?
 more: 6.6484375
 less: 6.0625
 rather: 5.21875
 slower: 4.734375
 greater: 4.671875

"""

#%%











