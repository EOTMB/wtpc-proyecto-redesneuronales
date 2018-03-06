from keras.models import Sequential
from keras.layers import SimpleRNN, TimeDistributed, Dense
import numpy as np

np.random.seed(1337)

sample_size = 256
x_seed = [1, 0, 0, 0, 0, 0]
y_seed = [1, 0.8, 0.6, 0, 0, 0]

x_train = np.array([[x_seed] * sample_size]).reshape(sample_size,len(x_seed),1)
y_train = np.array([[y_seed]*sample_size]).reshape(sample_size,len(y_seed),1)

model=Sequential()
model.add(SimpleRNN(input_dim  =  1, output_dim = 50, return_sequences = True))
model.add(TimeDistributed(Dense(output_dim = 1, activation  =  "sigmoid")))
model.compile(loss = "mse", optimizer = "rmsprop")
model.fit(x_train, y_train, nb_epoch = 10, batch_size = 32)

print(model.predict(np.array([[[1],[0],[0],[0],[0],[0]]])),x_train,y_train)
