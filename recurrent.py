import numpy as np
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

import matplotlib.pyplot as plt

##
# Data Set
##

set_dat = np.loadtxt('Valores_de_s.csv')
reset_dat = np.loadtxt('Valores_de_r.csv')
output_dat = np.loadtxt('Valores_de_o.csv')
'''
# Plot triple para visualizar
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(set_dat[0:1000], '-o', color = 'green', label = 'set')
ax2.plot(reset_dat[0:1000], '-o', color = 'blue', label = 'reset')
ax3.plot(output_dat[0:1000], '-o', color = 'red', label = 'output')
ax1.legend(numpoints = 1)
ax2.legend(numpoints = 1)
ax3.legend(numpoints = 1)
plt.ylim([-0.1,1.1])
plt.title('Batch simple')
plt.show()
'''
##
# Parametros de la red
# dim_inp es el tamano de inputs de la red
# dim_rnn es el de las capas internas de la red recurrente
# dim_out el tamano de la salida
##

dim_inp = 2
units_rnn = 50
dim_out = 1

##
# stateful: Boolean (default False). 
# If True, the last state for each sample at index i in a batch 
# will be used as initial state for the sample of index i in the 
# following batch.
#
# If a RNN is stateful, it needs to know its batch size. 
# Specify the batch size of your input tensors: 
# - If using a Sequential model, specify the batch size by passing 
#   a `batch_input_shape` argument to your first layer.
##

##
# ERROR QUE TIRA
# UserWarning: RNN dropout is no longer supported with the Theano 
# backend due to technical limitations. You can either set 
# `dropout` and `recurrent_dropout` to 0, or use the TensorFlow 
# backend. RNN dropout is no longer supported with the Theano backend 
##

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


##
# Formato de input de la red. CUIDADO, las redes neuronales recursivas se manejan con
# "timesteps", con lo cual la data es separada en pedazos. Ver el siguiente link:
# https://stackoverflow.com/questions/38294046/simple-recurrent-neural-network-input-shape
# 
##

##
# Hay que pensar en "seeds" de data y que le son entregados a la red en "timesteps"
# Shape del input de la red, es decir, de x_train: (largo_seed, cantidad_de_timesteps, dimension_de_input)
# Notar que aca, por ejemplo, como tenemos input de dimension 2 y el largo de la seed es de 25 vamos a tener 
# guardados arrays de shape (2, 25)
##

largo_seed = 10000
cantidad_de_timesteps = 100
entrada_array = np.zeros((largo_seed, cantidad_de_timesteps, dim_inp))
salida_array = np.zeros((largo_seed, cantidad_de_timesteps, dim_out))
print('entrada_array shape '+str(np.shape(entrada_array)))
for i in range(cantidad_de_timesteps):
    set_seed = set_dat[i * largo_seed : i * largo_seed + largo_seed]
    reset_seed = reset_dat[i : i + largo_seed]
    entrada_array[:,i,0] = set_seed
    entrada_array[:,i,1] = reset_seed
    out_seed = output_dat[i : i + largo_seed]
    salida_array[:,i,0] = out_seed


x_train = entrada_array
y_train = salida_array

print('------------------------------------------------------------------')
print('Formatos de los datos de entrada a la red y al fit')
print('Shape array de entrada: ' + str(np.shape(entrada_array)))
print('Shape datos de entrada del fit: ' + str(np.shape(x_train)))
print('Shape array de salida: ' + str(np.shape(salida_array)))
print('Shape datos de salida del fit: ' + str(np.shape(y_train)))
print('------------------------------------------------------------------')

# Paso de compilacion
# For a mean squared error regression problem
# model.compile(optimizer='rmsprop',
#               loss='mse')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'], sample_weight_mode = "temporal")

# Train the model, iterating on the data in batches
history = model.fit(x_train, y_train,
                    epochs=10, batch_size = 25)


# Test simple
test_s = set_dat[0:1000]
test_r = reset_dat[0:1000]
test_o = output_dat[0:1000]
test_in = np.zeros((1000,1,2))
test_in[:,0,0] = test_s
test_in[:,0,1] = test_r
test_out = np.zeros((1000,1,1))
test_out[:,0,0] = test_o

out = np.array(model.predict(test_in))
out = out[:,0,0]
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

