#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.layers import Embedding, GRU, Dense, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from nltk import word_tokenize, sent_tokenize
import keras.utils as ku 
import pandas as pd
import numpy as np
import string, os 
import warnings
import jieba
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


lyrics = pd.read_csv("JayChou2.csv", usecols=['lyric_full'])
stop_sign = ['[', "'", ']', '，', ' ']

lyrics = lyrics.values.tolist()
lyrics_main = []
lyrics_chorus = []
lyrics[0]


# In[3]:


for lines in lyrics:
    para = lines[0].split('。')
    for i in range(len(para)):
        para_clear = []
        for char in para[i]:
            if char not in stop_sign:
                para_clear.append(char)
        para_clear = ''.join(para_clear)
        if len(para) > 4:
            if i % 2 == 0 and i != 0:
                lyrics_chorus.append(para_clear)
            else:
                lyrics_main.append(para_clear)
        else:
            if len(para) > 2:
                if i == len(para) - 1:
                    lyrics_chorus.append(para_clear)
                else:
                    lyrics_main.append(para_clear)
            else:
                lyrics_main.append(para_clear)
        
len(lyrics_main)


# In[4]:


lyrics_main_cut = []
for lines in lyrics_main:
    lines = jieba.lcut(lines)
    lyrics_main_cut.append(lines)
lyrics_chorus_cut = []
for lines in lyrics_chorus:
    lines = jieba.lcut(lines)
    lyrics_chorus_cut.append(lines)


# In[5]:


lyrics_chorus_cut_flat = [num for sublist in lyrics_chorus_cut for num in sublist]
lyrics_main_cut_flat = [num for sublist in lyrics_main_cut for num in sublist]
len(lyrics_main_cut_flat)


# In[6]:


lyrics_total = lyrics_main_cut_flat + lyrics_chorus_cut_flat
len(lyrics_total)


# In[7]:


seg_len = 5

segments_main = []
next_chars_main = []
for j in range(0, len(lyrics_main_cut_flat) - seg_len):
    segments_main.append(lyrics_main_cut_flat[j:j + seg_len])
    next_chars_main.append(lyrics_main_cut_flat[j + seg_len])
segments_main[200]


# In[8]:


#construct one-hot vectors
vocab_main = sorted(list(set(lyrics_main_cut_flat)))
vocab_main_size = len(vocab_main)

segments_vector_main = np.zeros((len(segments_main), seg_len, vocab_main_size), int)
next_chars_vector_main = np.zeros((len(segments_main), vocab_main_size), int)
for i, segment in enumerate(segments_main):
    for j, char in enumerate(segment):
        segments_vector_main[i, j, vocab_main.index(char)] = 1
    next_chars_vector_main[i, vocab_main.index(next_chars_main[i])] = 1


# In[9]:


#GRU model
model1 = Sequential()
model1.add(GRU(16, input_shape=(seg_len, vocab_main_size)))
model1.add(Dense(vocab_main_size, activation='softmax'))
model1.summary()


# In[10]:


from keras.optimizers import Adam
optimizer =  Adam()
model1.compile(loss='categorical_crossentropy', optimizer=optimizer)

model1.fit(segments_vector_main, next_chars_vector_main, batch_size=5, epochs=8)


# In[11]:


segments_chorus = []
next_chars_chorus = []
for j in range(0, len(lyrics_chorus_cut_flat) - seg_len):
    segments_chorus.append(lyrics_chorus_cut_flat[j:j + seg_len])
    next_chars_chorus.append(lyrics_chorus_cut_flat[j + seg_len])
segments_chorus[200]


# In[12]:


#construct one-hot vectors
vocab_chorus = sorted(list(set(lyrics_chorus_cut_flat)))
vocab_chorus_size = len(vocab_chorus)

segments_vector_chorus = np.zeros((len(segments_chorus), seg_len, vocab_chorus_size), int)
next_chars_vector_chorus = np.zeros((len(segments_chorus), vocab_chorus_size), int)
for i, segment in enumerate(segments_chorus):
    for j, char in enumerate(segment):
        segments_vector_chorus[i, j, vocab_chorus.index(char)] = 1
    next_chars_vector_chorus[i, vocab_chorus.index(next_chars_chorus[i])] = 1


# In[13]:


#GRU model
model2 = Sequential()
model2.add(GRU(16, input_shape=(seg_len, vocab_chorus_size)))
model2.add(Dense(vocab_chorus_size, activation='softmax'))
model2.summary()


# In[14]:


from keras.optimizers import Adam
optimizer =  Adam()
model2.compile(loss='categorical_crossentropy', optimizer=optimizer)

model2.fit(segments_vector_chorus, next_chars_vector_chorus, batch_size=5, epochs=8)


# In[15]:


def sample(pred, temperature):
    pred = pred ** (1 / temperature)
    pred = pred / np.sum(pred)
    return np.argmax(pred)


# In[16]:


temperature_list = [0.1, 0.3, 0.5, 0.7, 0.9]


# In[ ]:


import random
import sys

keyword = []
answer_sen = []
text_len = 10
for time in range(300):
    target_index = random.randint(0, len(lyrics) - 1)
    generated_text = lyrics[target_index][0][:5]
    print(time)

    # generated_text = '你還在說是' # 讀入方式可以再想想
    generated_text_cut = jieba.lcut(generated_text)
    for i in range(len(temperature_list)):
        try:
            answer = generated_text
            temperature = temperature_list[i]
            next_anal = generated_text_cut
            for k in range(5):
                if k != 2 or k != 4:
                    while len(answer) < text_len * (k + 1):
                        sampled = np.zeros((1, seg_len, vocab_main_size), int)
                        for j, word in enumerate(next_anal):
                            sampled[0, j, vocab_main.index(word)] = 1
                        pred = model1.predict(sampled, verbose=0)[0]
                        next_index = sample(pred, temperature)
                        next_char = vocab_main[next_index]
                        answer += str(next_char)
                        next_anal.append(next_char)
                        next_anal.pop(0)
                else:
                    while len(answer) < text_len * (k + 1):
                        sampled = np.zeros((1, seg_len, vocab_main_size), int)
                        for j, word in enumerate(next_anal):
                            sampled[0, j, vocab_chorus.index(word)] = 1
                        pred = model2.predict(sampled, verbose=0)[0]
                        next_index = sample(pred, temperature)
                        next_char = vocab_chorus[next_index]
                        answer += str(next_char)
                        next_anal.append(next_char)
                        next_anal.pop(0)
                        print(next_anal)
#             sys.stdout.write(answer)
#             sys.stdout.write("\n")
            if temperature == 0.5:
                keyword.append(''.join(generated_text))
                answer_sen.append(answer)
        except:
            continue


# In[28]:


import csv
with open('answerStat.csv', mode='w', encoding='utf-8') as outcome:
    writer = csv.writer(outcome)
    writer.writerow(["keyword","output with temp = 0.5"])
    for i in range(len(keyword)):
        writer.writerow([keyword[i], answer_sen[i]])


# In[ ]:




