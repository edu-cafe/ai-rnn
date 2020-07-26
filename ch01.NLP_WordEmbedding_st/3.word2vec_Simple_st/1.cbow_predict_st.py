# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 23:24:05 2020

@author: shkim
"""

#%%
# 추론 기반 기법, 신경망으로 단어를 처리
import numpy as np

c = np.array([[1,0,0,0,0,0,0]])  # input(context) : One-hot Encoding
import numpy as np
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
## word2vec - CBOW 모델의 추론 처리
* 입력층이 2개인 이유는 맥락으로 고려할 단어를 2개로 정했기 때문임
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
in_layer0 = .....
in_layer1 = .....
out_layer = .....

# 순전파 
h0 = .....
h1 = .....
h = .....
s = .....

print(s)
# [[2.12397607 1.51623827 1.55780322 1.34659255 0.81729941 0.95288656 1.00042243]]

#%%




