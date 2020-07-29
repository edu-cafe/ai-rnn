# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 23:24:05 2020

@author: shkim
"""

"""
## 단순한 word2vec
* word2vec이라는 용어는 원래 프로그램이나 도구를 가리키는 데 사용됨
* 이용어가 유명해지면서, 문맥에 따라서는 신경망 모델을 가리키는 경우도 많이 볼 수 있음
* CBOW모델과 skip-gram 모델은 word2vec에서 사용되는 신경망임

## CBOW(Continuous bag-of-words) 모델
* CBOW 모델은 맥락으로부터 타깃(target)을 추측하는 용도의 신경망임 
* 타깃은 중안 단어이고 그 주변 단어들이 맥락임
* CBOW 모델이 가능한 한 정확하게 추론하도록 훈련시켜서 단어의 분산 표현을 얻어낼 것임
"""

#%%
# 추론 기반 기법, 신경망으로 단어를 처리
import numpy as np

c = np.array([[1,0,0,0,0,0,0]])  # input(context) : One-hot Encoding
W = np.random.rand(7, 3)  # weight
h = np.matmul(c, W)  # hidden node
print(h)  # [[0.80897103 0.76971805 0.67606468]]
# c와 W의 행렬곱은 결국 가중치의 행벡터 하나를 뽑아내는 것과 같음

#%%
# 추론 기반 기법, 신경망으로 단어를 처리
import sys
sys.path.append('../../')
from myutils.layers import MatMul

c = np.array([[1,0,0,0,0,0,0]])  # input(context) : One-hot Encoding
W = np.random.rand(7, 3)  # weight
layer = MatMul(W)
h = layer.forward(c)  
print(h)  # [[0.15648314 0.67678783 0.38569244]]

#%%
"""
## CBOW 모델의 추론 처리
* 입력층이 2개인 이유는 맥락으로 고려할 단어를 2개로 정했기 때문임
  맥락에 포함시킬 단어가 N개라면 입력층도 N개가 됨 
* 은닉층 뉴런 수를 입력층의 뉴런 수보다 적게 하는 것이 중요한 핵심임
* 이렇게 해야 은닉층에는 단어 예측에 필요한 정보를 간결하게 담게되며, 
  결과적으로 밀집벡터 표현을 얻을 수 있음
* 은닉층 : 인코딩, 촐력층 : 디코딩
"""
import numpy as np
import sys
sys.path.append('..')
from myutils.layers import MatMul

# 샘플 맥락 데이터
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])

# 가중치 초기화 
W_in = np.random.rand(7, 3)
W_out = np.random.rand(3, 7)

# 계층 생성
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파 
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

print(s)
# [[2.12397607 1.51623827 1.55780322 1.34659255 0.81729941 0.95288656 1.00042243]]

#%%




