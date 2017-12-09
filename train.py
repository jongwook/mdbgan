import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras.optimizers import RMSprop

from data import *

gen = stft_generator()

model = Sequential()
model.add(Dense(512, activation='tanh', input_shape=(513,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer=RMSprop(), metrics=['mae'])
history = model.fit_generator(gen, steps_per_epoch=10000, epochs=1000, verbose=1)
