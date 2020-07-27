# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:44:48 2020

@author: shkim
"""

"""
## RNN Language Model 구현 
* myutils.time-layers : TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss, TimeDropout
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.time_layers import TimeRNN, TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss

#%%
class SimpleRnnLm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # Initialize Weights
        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')  # Xavier Initialize
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')  # Xavier Initialize
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')  # Xavier Initialize
        affine_b = np.zeros(V).astype('f')
        
        # Create Layers
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
            ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]
        
        # Aggregate all Weights and Gradients
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.rnn_layer.reset_state()

#%%
"""
## 언어 모델의 평가
* 언어 모델은 주어진 과거 단어(정보)로부터 다음에 출현할 단어의 확률분포를 출력함
* 이때 언어 모델의 예측 성능을 평가하는 척도로 **'퍼플렉서티(perplexity,혼란도)'**를 자주 이용함
* perplexity 값이 작을 수록 모델의 예측 성능이 좋음 
* perplexity = e^L  // L : Cross-Entropy Error, 입력데이터가 여러 개일때 수식
* 정보이론 분야에서는 perplexity를 '기하평균 분기 수'라고도 함

### 퍼플렉서티(perplexity, 혼란도)
* perplexity는 **'확률의 역수'**임
* 언어모델1에 'you'를 입력 했을때 출력확률분포 중 'say'가 0.8이고 답이 'say'인경우
  perplexity = 1/0.8 = 1.25임
* 언어모델2에 'you'를 입력 했을때 출력확률분포 중 'say'가 0.2이고 답이 'say'인경우
  perplexity = 1/0.2 = 5임
* perplexity는 작은 것이 좋다는 것을 알 수 있음

### perlexity 수치에 대한 해석
* 이 값은 **'분기수(number of branches)'**로 해석할 수 있음
* 분기수란 다음에 취할 수 있는 선택사항의 수(다음에 출현할 수 있는 단어의 후보 수)를 말함
* 분기수가 1.25라는 것은 다음에 출현할 수 있는 단어의 후보를 1개 정도로 좁혔다는 뜻
* 분기수가 5라는 것은 다음에 출현할 수 있는 단어의 후보가 아직 5개나 된다는 뜻

* myutils.util --> eval_perplexity()
"""

#%%
"""
## RNN Languge Model Training 코드 구현
* PTB 데이터셋을 이용하여 RNN 언어 모델을 학습시켜보자 
* PTB 데이터셋의 일부(1000개 단어)만 사용할 것임 
  --> 아직 성능이 나오지 않은 모델이기 때문 --> 다음 코드에서 개선할 것임(LSTM, GRU)
  
## RNN Language Model Training 순서
* 1) 미니배치를 순차적으로 만들어
* 2) 모델의 순전파와 역전파를 호출하고
* 3) 옵티마이저로 가중치를 갱신하고
* 4) perplexity를 구함
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.optimizer import SGD
from ptb_dataset import load_data

#%%
# Setting Hyperparameters
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN 은닉 상태 벡터의 원소 수
time_size = 5  # Truncated BPTT가 한번에 펼치는 시간 크기 
lr = 0.1
max_epoch = 100

# Read Training Data : 전체 데이터 중 1000개
corpus, word_to_id, id_to_word = load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
# print('corpus:', corpus[500:510])  # [241 242  42  61  26 243 108 244 172  48]
# print('vocab_size:', vocab_size)  # 418

xs = corpus[:-1]  # input
ts = corpus[1:]  # label
data_size = len(xs)
print('말뭉치 크기:%d, 어휘수:%d, 입력 데이터 크기:%d' % 
      (corpus_size, vocab_size, data_size))  # 1000, 418, 999

#%%
# Variables for training
max_iters = data_size // (batch_size * time_size)  # 19
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []  # perplexity list

# Create Model
model = SimpleRnnLm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

#%%
# 1) 각 미니배치에서 읽을 샘플의 위치 계산 
jump = (corpus_size - 1) // batch_size  # 99
offsets = [i*jump for i in range(batch_size)]  # [0, 99, 198, 297, 396, 495, 594, 693, 792, 891]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 2) 미니배치 획득
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        # 3) 기울기를 구하여 매개변수 갱신
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1
    
    # 4) epoch마다 perplexity 계산
    ppl = np.exp(total_loss / loss_count)
    print('-->epoch:%d, peplexity:%.2f' % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

"""
-->epoch:1, peplexity:395.05
-->epoch:2, peplexity:267.11
-->epoch:3, peplexity:224.26
-->epoch:4, peplexity:215.44
-->epoch:5, peplexity:205.21
-->epoch:6, peplexity:201.94
-->epoch:7, peplexity:197.95
-->epoch:8, peplexity:196.80
-->epoch:9, peplexity:191.14
-->epoch:10, peplexity:192.40
-->epoch:11, peplexity:189.00
-->epoch:12, peplexity:192.77
-->epoch:13, peplexity:190.79
-->epoch:14, peplexity:191.09
-->epoch:15, peplexity:190.35
-->epoch:16, peplexity:186.18
-->epoch:17, peplexity:183.93
-->epoch:18, peplexity:181.64
-->epoch:19, peplexity:182.24
-->epoch:20, peplexity:182.99
-->epoch:21, peplexity:182.93
-->epoch:22, peplexity:178.97
-->epoch:23, peplexity:175.60
-->epoch:24, peplexity:177.41
-->epoch:25, peplexity:173.43
-->epoch:26, peplexity:174.07
-->epoch:27, peplexity:169.22
-->epoch:28, peplexity:168.27
-->epoch:29, peplexity:164.24
-->epoch:30, peplexity:160.54
-->epoch:31, peplexity:162.23
-->epoch:32, peplexity:157.09
-->epoch:33, peplexity:157.92
-->epoch:34, peplexity:151.64
-->epoch:35, peplexity:150.20
-->epoch:36, peplexity:143.58
-->epoch:37, peplexity:139.03
-->epoch:38, peplexity:138.74
-->epoch:39, peplexity:130.57
-->epoch:40, peplexity:130.31
-->epoch:41, peplexity:127.64
-->epoch:42, peplexity:119.90
-->epoch:43, peplexity:114.57
-->epoch:44, peplexity:111.48
-->epoch:45, peplexity:107.71
-->epoch:46, peplexity:105.47
-->epoch:47, peplexity:101.15
-->epoch:48, peplexity:93.80
-->epoch:49, peplexity:92.87
-->epoch:50, peplexity:90.98
-->epoch:51, peplexity:85.63
-->epoch:52, peplexity:80.25
-->epoch:53, peplexity:75.88
-->epoch:54, peplexity:72.93
-->epoch:55, peplexity:69.73
-->epoch:56, peplexity:66.29
-->epoch:57, peplexity:61.46
-->epoch:58, peplexity:59.83
-->epoch:59, peplexity:57.02
-->epoch:60, peplexity:52.13
-->epoch:61, peplexity:51.13
-->epoch:62, peplexity:47.70
-->epoch:63, peplexity:44.62
-->epoch:64, peplexity:41.94
-->epoch:65, peplexity:40.31
-->epoch:66, peplexity:38.14
-->epoch:67, peplexity:36.30
-->epoch:68, peplexity:32.83
-->epoch:69, peplexity:31.98
-->epoch:70, peplexity:30.41
-->epoch:71, peplexity:28.40
-->epoch:72, peplexity:27.63
-->epoch:73, peplexity:26.26
-->epoch:74, peplexity:24.26
-->epoch:75, peplexity:23.04
-->epoch:76, peplexity:21.11
-->epoch:77, peplexity:19.67
-->epoch:78, peplexity:18.49
-->epoch:79, peplexity:17.47
-->epoch:80, peplexity:16.74
-->epoch:81, peplexity:16.01
-->epoch:82, peplexity:15.25
-->epoch:83, peplexity:14.23
-->epoch:84, peplexity:13.41
-->epoch:85, peplexity:12.74
-->epoch:86, peplexity:12.22
-->epoch:87, peplexity:11.69
-->epoch:88, peplexity:10.78
-->epoch:89, peplexity:9.53
-->epoch:90, peplexity:9.49
-->epoch:91, peplexity:9.22
-->epoch:92, peplexity:8.49
-->epoch:93, peplexity:8.24
-->epoch:94, peplexity:7.99
-->epoch:95, peplexity:7.32
-->epoch:96, peplexity:6.85
-->epoch:97, peplexity:6.84
-->epoch:98, peplexity:6.74
-->epoch:99, peplexity:6.16
-->epoch:100, peplexity:5.93
"""
#%%
# 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
        
#%%












            
        
        









