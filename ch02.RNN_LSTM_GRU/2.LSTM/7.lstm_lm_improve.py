# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:12:22 2020

@author: shkim
"""

"""
# LSTM Language Model 성능 개선
* 1) LSTM 계층 다층화 
* 2) Dropout에 의한 과적합 억제
* 3) 가중치 공유
"""

"""
## 1) LSTM 계층 다층화 
* LSTM LM으로 정확한 모델을 만들고자 한다면 많은 경우 LSTM 계층을 깊게 쌓아(계층을 여러 겹 쌓아) 효과를 볼 수 있음
* LSTM 계층을 2층, 3층 식으로 여러 겹 쌓으면 언어 모델의 정확도가 향상되리라 기대할 수 있음
* LSTM 계층을 몇 층이라도 쌓을 수 있으며, 그 결과 더 복잡한 패턴을 학습할 수 있게 됨
* PTB 데이터 셋의 언어 모델에서는 LSTM의 층수는 2~4 정도일 때 좋은 결과를 얻는 것 같음
* 구글 번역에 사용하는 GNMT 모델은 LSTM을 8층 쌓은 신경망이라고 함
* 처리할 문제가 복잡하고 학습 데이터를 대량으로 준비할 수 있다면 LSTM 층을 깊게 쌓는 것이 정확도 향상을 이끌 수 있음
"""

"""
## 2) Dropout에 의한 과적합 억제
* LSTM 층을 깊게 쌓음으로써 표현력이 풍부한 모델을 만들 수 있음
* 그러나 이런 모델은 종종 과적합(overfitting)을 일으킴 
* 불행하게도 RNN(LSTM)은 일반 피드포워드 신경망보다 쉽게 과적합을 일으키는 것으로 알려짐 
* 따라서 RNN(LSTM)의 과적합 대책은 중요하고, 현재도 홯발히 연구되는 주제임 

### 과적합을 억제하는 전통적인 방법
* 훈련 데이터의 양 늘리기
* 모델의 복잡도 줄이기
* 정규화(Normalization)
* Regularization(제약) : L2 Regilarization은 가중치가 너무 커지면 패널티를 부과함 
* Dropout : 무작위로 뉴런을 선택해서 선택한 뉴런을 무시함
  - 무시한다는 말은 그 앞 계층으로부터의 신호 전달을 막는다는 뜻임 
  - '무작위한 무시'가 제약이 되어 신경망의 일반화 성능을 개선하는 것임
"""

"""
## 3) 가중치 공유 (Weight Tying)
* 언어 모델을 개선하는 아주 간단한 트릭 중 Weight Tying 기법이 있음
* Weigt Tying을 직역하면 '가중치를 연결한다'이지만, 실질적으로는 가중치를 공유하는 효과를 줌
* Embedding 계층의 가중치와 Affine 계층의 가중치를 연결(공유)하는 기법이 Weight Tying 임
* 두 계층이 가중치를 공유하으로써 학습하는 매개변수 수가 크게 줄어드는 동시에 정확도도 향상되는 일석이조의 기술임 
* 어휘수를 V, LSTM 은닉상태의 차원 수를 H라고 하면
  - Embedding 계층의 가중치 형상은 V x H, Affine 계층의 가중치 형상은 H x V
  - 가중치 공유를 적용하려면, Embedding 계층의 가중치를 전치하여 Affine 계층의 가중치로 설정하기만 하면됨 
  
* 가중치를 공유함으로써 학습해야 할 매개변수 수를 줄일 수 있고, 그 결과 학습하기가 더 쉬워짐 
* 게다가 매개변수 수가 줄어든다는 것은 과적합이 억제되는 혜택으로 이어질 수 있음 
"""

#%%
"""
## 개선된 LSTM Language Model 구현 

* myutils.time_layers --> BetterLstmLm Class
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.time_layers import TimeEmbedding, TimeDropout, TimeLSTM, TimeAffine, TimeSoftmaxWithLoss
from myutils.base_model import BaseModel

#%%
class BetterLstmLm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=650, 
                 hidden_size=650, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # Initialize Weights
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4*H) / np.sqrt(D)).astype('f')  # Xavier Initialize
        lstm_Wh1 = (rn(H, 4*H) / np.sqrt(H)).astype('f')  # Xavier Initialize
        lstm_b1 = np.zeros(4*H).astype('f')
        lstm_Wx2 = (rn(D, 4*H) / np.sqrt(H)).astype('f')  # Xavier Initialize
        lstm_Wh2 = (rn(H, 4*H) / np.sqrt(H)).astype('f')  # Xavier Initialize
        lstm_b2 = np.zeros(4*H).astype('f')
        affine_b = np.zeros(V).astype('f')
        
        # 세 가지 개선 
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)  # Weight Tying
            ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
        
        # Aggregate all Weights and Gradients
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
    
#%%





