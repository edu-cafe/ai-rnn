# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:21:07 2020

@author: shkim
"""

"""
# 언어 모델(Language Model) 개요 
* 언어 모델은 단어 나열에 확률을 부여함
* 특정한 단어의 시퀀스에 대해서 그 시퀀스가 일어날 가능성이 어느 정도인지
  (얼마나 자연스러운 단어 순서인지)를 확률로 평가하는 것임
* 'you say goodbye'시퀀스가 'you say good die'시퀀스보다 높은 확률을 출력하는 모델
* 이 언어 모델은 다양하게 응용할 수 있음 : 기계번역, 음성인식 등
"""

"""
## 언어 모델의 수학적 표현
* 동시확률 P(W1,...,Wm-1,Wm) = 파이_t=1~m P(Wt | W1,...,Wt-1)
* 동시 확률은 사후 확률의 총곱으로 나타낼 수 있음
* P(W1,...,Wm-1,Wm) = P(A,Wm) = P(Wm|A)P(A)
* P(W1,...,Wm-2,Wm-1) = P(A',Wm-1) = P(Wm-1|A')P(A')
* 이 사후 확률은 타깃 단어보다 왼쪽에 있는 모든 단어를 맥락(조건)으로 했을 때의 확률임
* 참고: 확률의 곱셈 정리 -->  P(A,B) = P(A|B)P(B) = P(B|A)P(A)

## 언어 모델이 다루는 사후 확률
* P(Wt|W1,...Wt-1) :조건부 언어 모델(Conditional Language Model)
* i번째 단어를 타깃으로 하여 i번째보다 왼쪽 단어 모두를 맥락(조건)으로 고려함 
"""

"""
## CBOW 모델을 언어 모델로 했을 때의 문제점
* CBOW 모델은 2층 마르코프 체인(혹은 마르코프 모델)으로 볼 수 있음
* 참고 : 마르코프 체인이란 미래의 상태가 현재의 상태에만 의존해 결정되는 것을 말함
* CBOW 모델을 N층 마르코프 모델로 만들더라도, CBOW 모델에서는 맥락 안의 단어 순서가 무시됨
* CBOW 모델의 은닉층에서는 단어 벡터들이 더해지므로 맥락의 단어 순서는 무시됨 
  (you, say)와 (say, you)라는 맥락은 똑같이 취급됨
"""

"""
## 맥락의 단어 순서도 고려한 모델이 바람직 할 것임!!
* 맥락의 단어 벡터를 은닉층에서 연결(concatenate)하는 방식을 생각할 수 있음
  --> 신경 확률론적 언어 모델(Neural Probabilistic Language Model)
* 그러나 연결하는 방식을 취하면 맥락의 크기에 비례해 가중치 매개변수도 늘어나는 문제가 있음
* 이 문제를 해결하는 모델이 RNN임!!
  --> RNN은 맥락이 아무리 길더라도 그 맥락의 정보를 기억하는 메커니즘을 갖추고 있음 
  --> RNN을 사용하면 아무리 긴 시계열 데이터라도 대응할 수 있음 
"""

"""
# RNN(Recurrent Nueral Network, 순환 신경망) 개요 
## 순환의 의미는?
* 어느 한 지점에서 시작한 것이 시간을 지나 다시 원래 장소로 돌아오는 것, 
  그리고 이 과정을 반복하는 것이 '순환'임 --> 순환을 위해서는 '닫힌 경로'가 필요함
* '닫힌 경로' 혹은 '순환하는 경로'가 존재해야 데이터가 같은 장소를 반복해 왕래할 수 있음
* 그리고 데이터가 순환하면서 정보가 끊임없이 갱신되게 됨
* 이는 우리 체내의 혈액에 비유할 수 있음 

## RNN의 특징은 순환하는 경로(닫힌 경로)가 있다는 것임
* 이 순환 경로를 따라 데이터는 끊임없이 순환할 수 있음
* 데이터가 순환되기 때문에 과거의 정보를 기억하는 동시에 최신 데이터로 갱신될 수 있음 

## RNN 현 시각의 출력을 계산하는 수식
* h_t = tanh(h_t-1*W_h + x_t*W_x + b)
* 현재의 출력(h_t)은 한 시각 이전 출력(h_t-1)에 기초해 계산됨을 알 수 있음
* RNN은 h라는 '상태'를 가지고 있으며, 이 상태는 위 식의 형태로 갱신됨
* 그래서 RNN 계층을 ***'상태를 가지는 계층'*** 혹은 ***'메모리가 있는 계층'***이라고 함
* h_t를 은닉 상태(Hidden State) 혹은 은닉 상태 벡터(Hidden State Vector)라고 함

## BPTT(BackPropagation Through Time)란?
* RNN에는 (일반적인) 오차역전파법을 적용할 수 있음
* 여기서의 오차역전파법은 '시간 방향으로 펼친 신경망의 오차역전파법'임 --> BPTT
* BPTT에서는 한 가지 문제가 있음 --> 긴 시계열 데이터를 학습할 때의 문제
* 시계열 데이터의 시간 크기가 커지는 것에 비례하여 BPTT가 소비하는 컴퓨팅 자원도 증가하기 때문
* 시간 크기가 커지면 역전파 시의 기울기가 불안정해지는 것도 문제임
* 시계열 데이터가 길어짐에 따라 계산량뿐 아니라 메모리 사용량도 증가하게 됨

## Truncated BPTT
* Truncated BPTT는 적당한 길이로 ***'잘라낸'*** 오차역전파법임
* 큰 시계열 데이터를 취급할 때는 흔히 신경망 연결을 적당한 길이로 끊음
* 주의할 사항은 역전파의 연결만 끊어야함, 순전파의 연결은 반드시 그대로 유지해야 함
* 순전파의 흐름은 끊어지지 않고 전파되고, 
* 역전파의 연결은 적당한 길이로 잘라내 그 잘라낸 신경망 단위로 학습을 수행함 
* 역전파가 연결되는 일련의 RNN 계층을 ***'블록'***이라 함
* Truncated BPTT에서 미니배치 학습을 수행할 때는 데이터를 ***'순서대로'*** 입력해야 함
* 길이 1000인 시계열 데이터를 한블럭이 10개 단위이고 2개의 미니배치로 학습을 시키는 경우
  첫번째 미니배치 원소는 (x0,x1,..,x9),(x10,x11,..,x19)...(x490,x491,..,x499)
  두번째 미니배치 원소는 (x500,x501,..,x509),(x510,x511,..,x519)...(x990,x991,..,x999)
*
"""
#%%
"""
## RNN Layer 구현
* myutils.time-layers
"""
import numpy as np

class RNN:
    # N : 미니배치 크기, H : 은닉 상태 벡터의 차원 수, D : 입력 벡터의 차원 수
	# Wx:(D, H), Wh:(H, H) 
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
    
    # x:(N, D), h_prev:(N, H) 
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)
        
        self.cache = (x, h_prev, h_next)
        return h_next
    
    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache
        
        dt = dh_next * (1 - h_next ** 2)  # tanh: 1-a**2
        db = np.sum(dt, axis=0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wx.T)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        return dx, dh_prev

#%%
"""
## Time RNN Layer 구현
* Time RNN 계층은 T개의 RNN 계층으로 구성됨 
* Time RNN 계층은 RNN 계층 T개를 연결한 신경망임 
* Time RNN 계층은 은닉 상태를 인스턴스 변수 h로 보관하여 은닉상태를 다음 블록에 인계할 수 있음
* myutils.time-layers
"""
import numpy as np

class TimeRNN:
    # stateful : 은닉상태를 인계 받을 것인지를 나타냄 (은닉 상태 유지 모드)
    # 긴 시계열 데이터를 처리할 때에는 RNN 은닉 상태를 유지해야 함 
    # Wx:(D, H), Wh:(H, H) 
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None  # save RNN Layers
        
        self.h, self.dh = None, None
        self.stateful = stateful
    
    def set_state(self, h):
        self.h = h
    
    def reset_state(self):
        self.h = None
    
    # xs shape : (N, T, D) N:미니배치의 수, T:T개 시간 단계분의 작업, D:입력벡터의 차원수
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # N:미니배치의 수, T:T개 시간 단계분의 작업, D:입력벡터의 차원수
        D, H = Wx.shape  # D:입력벡터의 차원수, H:은닉벡터의 차원수
        
        self.layers = []
        hs = np.empty((N,T,H), dtype='f')
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H), dtype='f')
        
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
            
        return hs
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        
        dxs = np.empty((N,T,D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)  # 합산된 기울기
            dxs[:, t, :] = dx
            
            for i, grade in enumerate(layer.grads):
                grads[i] += grade
            
        for i, grade in enumerate(grads):
            self.grads[i][...] = grade
        self.dh = dh
        
        return dxs


#%%







