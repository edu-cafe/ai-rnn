# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:22:08 2020

@author: shkim
"""

"""
## LSTM Language Model 구현 

* myutils.time-layers
"""
import numpy as np
import sys
sys.path.append('../../')
from myutils.time_layers import TimeLSTM, TimeEmbedding, TimeAffine, TimeSoftmaxWithLoss
import pickle

#%%
class LstmLm:
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # Initialize Weights
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')  # Xavier Initialize
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')  # Xavier Initialize
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')  # Xavier Initialize
        affine_b = np.zeros(V).astype('f')
        
        # Create Layers
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
            ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]
        
        # Aggregate all Weights and Gradients
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()
    
    def save_params(self, file_name='LstmLm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
        
    def load_params(self, file_name='LstmLm.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)

#%%
