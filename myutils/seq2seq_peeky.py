# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 11:24:24 2020

@author: shkim
"""

"""
## seq2seq 성능 개선
* 1) 입력 데이터 반전(Reverse)
* 2) 엿보기(Peeky)
"""

"""
## seq2seq 성능 개선 : 엿보기(Peeky)
### base seq2seq 모델에서의 동작
* Encoder는 입력문장(문제문장)을 고정길이 벡터 h로 변환함
* 이때 h안에는 Decoder에 필요한 정보가 모두 담겨 있음
* 즉 h가 Decoder에 있어서 유일한 정보인 셈임
* 최초 시각의 LSTM 계층만이 벡터 h를 이용함 --> 이 중요한 h 정보를 더 활용할 수 없을까?

### 개선된 seq2seq 모델 : 엿보기(Peeky) 모델
* 중요한 정보가 담긴 Encoder의 출력 h를 Decoder의 다른 계층에도 전달해 주는 것
* Encoder의 출력 h를 모든 시각의 LSTM 계층과 Affine 계층에 전해줌 --> 집단 지성
* LSTM 계층과 Affine 계층에 입력되는 벡터가 2개씩 됨 --> concatenate 됨
"""

#%%
"""
## 개선된 seq2seq 모델 : 엿보기(Peeky) 모델 구현
"""

import numpy as np
import sys
sys.path.append('..')
from myutils.time_layers import TimeEmbedding, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
from myutils.seq2seq import Seq2seq, Encoder

#%%
class DecoderPeeky:
    def __init__(self, vocab_size, wordvec_size, hideen_size):
        V, D, H = vocab_size, wordvec_size, hideen_size
        rn = np.random.randn
        
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H+D, 4*H) / np.sqrt(H+D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H+H, V) / np.sqrt(H+H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)
        
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        
        self.cache = None
        
    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape
        
        self.lstm.set_state(h)
        
        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)
        
        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)
        
        score = self.affine.forward(out)
        self.cache = H
        return score
    

    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh


    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled

    
#%%   
class Seq2seqPeeky(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = DecoderPeeky(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
    
#%%





