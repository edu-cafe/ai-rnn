# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:11:11 2020

@author: shkim
"""

"""
## Gradients Clipping 구현 
* Gradients Exploding을 막는 전통적인 기법 --> Gradients Clipping
* if ||g^|| >= threshold:
    g^ = (threshold/||g^||) * g^
    here, g^은 신경망에서 사용되는 모든 매개변수의 기울기를 하나로 모은 것임 

* myutils.util
"""

import numpy as np

dW1 = np.random.rand(3, 3) * 10
dW2 = np.random.rand(3, 3) * 10
grads = [dW1, dW2]
max_norm = 5.0

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad**2)
    total_norm = np.sqrt(total_norm)
    
    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate
 
#%%
print('-->grads_before-clip: \n', grads)
clip_grads(grads, max_norm)
print('-->grads_after-clip: \n', grads)

"""
-->grads_before-clip: 
 [array([[5.57616591, 5.76718907, 7.83441125],
       [7.62043529, 3.6762898 , 8.26333283],
       [9.27802202, 0.26510388, 7.67360492]]), 
 array([[7.70127632, 2.87466644, 1.20299569],
       [9.64671631, 4.04126435, 1.99226532],
       [3.07090783, 3.30237996, 5.08161404]])]
-->grads_after-clip: 
 [array([[1.10279605, 1.14057462, 1.5494083 ],
       [1.50709037, 0.72705833, 1.63423595],
       [1.83491061, 0.05242949, 1.5176057 ]]), 
 array([[1.52307826, 0.5685216 , 0.23791596],
       [1.90782713, 0.79923919, 0.3940095 ],
       [0.60733218, 0.65311033, 1.00498873]])]
"""

