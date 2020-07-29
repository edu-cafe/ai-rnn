# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 20:00:43 2020

@author: shkim
"""

"""
# seq2seq
* 시계열 데이터 예 : 언어 데이터, 음성 데이터, 동영상 데이터 등 
* 시계열 데이터를 또다른 시계열 데이터로 변환하는 문제 : 기계번역, 음성인식, 쳇봇, 컴파일러 등
* 시계열 데이터를 다른 시계열 데이터로 변환하는 모델 : seq2seq
* seq2seq(sequence to sequence)는 2개의 RNN(LSTM)을 이용함 

## seq2seq 의 원리
* seq2seq를 Encoder-Decoder 모델이라고도 함
* Encoder는 입력 데이터를 인코딩(부호화)하고, Decoder는 인코딩된 데이터를 디코딩(복호화)함 
* 기계번역 : 한국어 --> 영어
  - Encoder가 '나는 고양이다'라는 출발 문장을 인코딩하고, 인코딩한 정보를 Decoder에게 전달하고
  - Decoder가 도착어 문장을 생성함
  - Encoder가 인코딩한 정보에는 번역에 필요한 정보가 조밀하게 응축되어 있음
* seq2seq는 Encoder와 Decoder가 협력하여 시계열 데이터를 다른 시계열 데이터로 변환하는 것임
  - Encoder가 출력하는 벡터 h는 LSTM 계층의 마지막 은닉 상태임, h는 고정길이 벡터임
  - 이 마지막 은닉 상태 h에 입력문장(출발어)을 번역하는 데 필요한 정보가 인코딩됨
  - 인코딩한다는 것은 임의 길이의 문장을 고정 길이 벡터로 변환하는 작업임 
  - Decoder는 Encoder가 생성한 벡터 h를 입력으로 받는 LSTM 신경망임

## 시계열 데이터 변환을 위한 Toy Problem
* 덧셈(addition) 계산 문제 --> Question & Answering Sentence
* dataset : addition.txt --> seq_dataset.py
"""

#%%
"""
## seq2seq 구현
* Encoder Class, Decoder Class, Seq2Seq Class
* myutils.seq2seq

## seq2seq - Encoder Class 구현
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.time_layers import TimeEmbedding, TimeLSTM

class Encoder:
    # vocab_size : 문자의 종류(0~9, '+', ' ', '_' 총 13가지 문자)
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        # statefule=False : 짧은 시계열 데이터가 여러 개인 문제이므로, 
        # 문제마다 LSTM 은닉상태를 다시 초기화한 상태(영벡터)로 설정함
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        
        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None
        
    # TimeLSTM 계층의 마지막 시각의 은닉상태만을 추출해 반환함
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]
    
    # dh : LSTM 계층의 마지막 은닉상태에 대한 기울기가 dh 인수로 전해짐
    # dh : Decoder가 전해준 기울기임 
    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh
        
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

#%%
"""
## seq2seq - Decoder Class 구현
* 문장 생성 시에는 소프트맥스 함수의 확률분포를 바탕으로 샘플링을 수행했기 때문에 
  생성되는 문장이 확률에 따라 달라졌음
* 이와 달리 이번 문제는 '덧셈'이므로 이러한 '비결정성'을 배제하고 '결정적'인 답을 생성해야함
* 이번 문제에서는 '확률적'이 아닌 '결정적'으로 선택할 것임 --> 'argmax' 노드 사용 
* 'argmax' 노드 : 최댓값을 가진 인덱스(문자의 ID)를 선택하는 노드 
  --> Affine 계층이 출력하는 점수가 가장 큰 문자 ID를 선택함
* Decoder 클래스는 TimeEmbedding, TimeLSTM, TimeAffine의 3가지 계층으로 구성됨

### Decoder 클래스는 학습 시와 문장 생성 시의 동작이 다름
* 학습 시에는 forward()의 메서드를 사용함
* 문장 생성 시에는 generate() 메서드를 사용함
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.time_layers import TimeEmbedding, TimeLSTM, TimeAffine

class Decoder:
    # vocab_size : 문자의 종류(0~9, '+', ' ', '_' 총 13가지 문자)
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)
        
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads        


    def forward(self, xs, h):
        self.lstm.set_state(h)
        
        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score
    
    # dscore : SoftmaxwithLoss 계층으로부터 기울기 dscore를 받음
    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh
    
    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)
        
        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)
            
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))
            
        return sampled


#%%
"""
## seq2seq - Seq2seq 클래스 구현
* Encoder 클래스와 Decoder 클래스를 연결하고, TimeSoftmaxWithLoss 계층을 이용해 손실을 계산함
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.time_layers import TimeSoftmaxWithLoss
from myutils.base_model import BaseModel

class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()
        
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
        
    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout
    
    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
    
#%%











