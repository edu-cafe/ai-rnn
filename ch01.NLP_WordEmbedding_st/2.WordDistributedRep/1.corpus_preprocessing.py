# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:28:40 2020

@author: shkim
"""

text = 'You say goodbye and I say hello.'
text = text.lower()
text = text.replace('.', ' .')
words = text.split(' ')
print(words)

#%%
word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word
        
print(word_to_id)
print(id_to_word)

#%%
import numpy as np

corpus = [ word_to_id[w] for w in words ]
corpus = np.array(corpus)
print(corpus)

#%%
# myutils.util
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
            
    corpus = [ word_to_id[w] for w in words ]
    
    return corpus, word_to_id, id_to_word

#%%
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(word_to_id)
print(id_to_word)

#%%






