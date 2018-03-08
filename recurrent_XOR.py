import numpy as np
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
from asd import *
from copia import *

##
# Data Set
##

largo_seed = 10000
cantidad_de_timesteps = 100

#x_train, y_train = trainingSRO(10,1000000, largo_seed, cantidad_de_timesteps, 2, 1)
x_train, y_train, mask = atanasov(largo_seed, cantidad_de_timesteps)

dim_inp = 2
units_rnn = 10
dim_out = 1

model = Sequential()
model.add(SimpleRNN(input_dim = dim_inp,
                    units = units_rnn,
                    stateful = False,
                    #batch_input_shape = (200000,10,2),
                    dropout = 0,
                    recurrent_dropout = 0,
                    return_sequences = True,
                    activation = "linear"))

# Que activacion usar ??
model.add(Dense(units = dim_out, activation = "linear"))


model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'], 
              sample_weight_mode = "temporal")


# Train the model, iterating on the data in batches
history = model.fit(x_train, y_train,
                    epochs=10, batch_size = 25,
                    sample_weight = mask)

# Test simple
test_s = x_train[0:1000,0,0]
test_r = x_train[0:1000,0,1]
test_o = y_train[0:1000,0,0]
test_in = np.zeros((1000,1,2))
test_in[:,0,0] = test_s
test_in[:,0,1] = test_r
test_out = np.zeros((1000,1,1))
test_out[:,0,0] = test_o

# La red parece estar siguiendo la siguiente regla
# array para chequear eso
xor = np.zeros(1000)
for idx in range(len(test_s)):
    if test_s[idx]== 0 and test_r[idx] == 0:
        xor[idx] = 0
    if test_s[idx]== 0 and test_r[idx] == 1:
        xor[idx] = 1
    if test_s[idx]== 1 and test_r[idx] == 0:
        xor[idx] = 1
    if test_s[idx]== 1 and test_r[idx] == 1:
        xor[idx] = 0

out = np.array(model.predict(test_in))
out = out[:,0,0]

plt.plot(xor, out, 'o')
plt.show()

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex = True, sharey = False)
ax1.plot(test_s, '-o', color = 'green', label = 'set real')
plt.ylim([-0.1,1.1])
ax2.plot(test_r, '-o', color = 'blue', label = 'reset real')
plt.ylim([-0.1,1.1])
ax3.plot(test_o, '-o', color = 'red', label = 'output real')
plt.ylim([-0.1,1.1])
ax4.plot(out, '-o', color = 'black', label = 'output red')
ax1.legend(numpoints = 1)
ax2.legend(numpoints = 1)
ax3.legend(numpoints = 1)
ax4.legend(numpoints = 1)
plt.ylim([-0.1,1.1])

plt.savefig('simple_test.png')
plt.show()

