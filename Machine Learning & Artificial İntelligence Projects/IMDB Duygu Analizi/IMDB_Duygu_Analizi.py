# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:49:27 2023
@author: Hasan Emre
"""

#%% import library

# Gerekli kütüphaneleri içe aktarir
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Embedding, Dense, Activation

#%% Load the IMDb dataset and split into training and test sets

# IMDb film incelemeleri veri kumesini yukler ve egitim ve test verilerini ayirir
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                     num_words=None,
                                                     skip_top=0,
                                                     maxlen=None,
                                                     seed=113,
                                                     start_char=1,
                                                     oov_char=2,
                                                     index_from=3)

# Egitim ve test verilerinin turunu ve seklini ekrana yazdirir
print("Type: ", type(x_train))
print("Type: ", type(y_train))

print("X train shape: ", x_train.shape)
print("Y train shape: ", y_train.shape)

#%% EDA (Exploratory Data Analysis)

# Egitim ve test verilerinin sinif dagilimlarini gorsellestirir
print("Y train values: ", np.unique(y_train))
print("Y test values: ", np.unique(y_test))

unique, counts =  np.unique(y_train, return_counts=True)
print("Y train distribution: ", dict(zip(unique, counts)))

unique, counts =  np.unique(y_test, return_counts=True)
print("Y test distribution: ", dict(zip(unique, counts)))

plt.figure()
sns.countplot(y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y train")

plt.figure()
sns.countplot(y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y test")

# Egitim verilerinden bir inceleme ornegini (film incelemesi) ve uzunlugunu ekrana yazdirir
d = x_train[0]
print(d)
print(len(d))

# Egitim ve test verilerinin inceleme uzunluklarinin dagilimini gorsellestirir
review_len_train = []
review_len_test = []

for i, ii in zip(x_train, x_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))

sns.distplot(review_len_train, hist_kws={"alpha": 0.3})
sns.distplot(review_len_test, hist_kws={"alpha": 0.3})

print("Train mean: ", np.mean(review_len_train))
print("Train median: ", np.median(review_len_train))
print("Train mode: ", stats.mode(review_len_train))

# IMDb veri kumesinin sozlugunu olusturur ve belirli bir kelimenin sozlukteki indeksini bulur
word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))

for keys, values in word_index.items():
    if values == 4:
        print(keys)

# Verilen indeksteki incelemenin icerigini ve etiketini ekrana yazdiran bir fonksiyon tanimlar
def whatItSay(index=24):
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in x_train[index]])
    print(decode_review)
    print(y_train[index])
    return decode_review

decoded_review = whatItSay(1000)

#%% Preprocess the data

# Veri kumesinin boyutunu azaltarak ve esit uzunluktaki incelemeleri doldurarak veri on isleme yapar
num_words = 15000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

maxlen = 130
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print(x_train[5])

for i in x_train[:10]:
    print(len(i))

decoded_review = whatItSay(5)

#%% Build and train the RNN model

# Basit RNN modelini olusturur ve egitir
rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length=len(x_train[0])))
rnn.add(SimpleRNN(16, input_shape=(num_words, maxlen), return_sequences=False, activation="relu"))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

print(rnn.summary())
rnn.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

history = rnn.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128, verbose=1)

# Modelin test verileri uzerinde performansini degerlendirir ve sonuclari ekrana yazdirir
score = rnn.evaluate(x_test, y_test)
print("Accuracy: %", score[1] * 100)

plt.figure()
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Test")
plt.title("Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
