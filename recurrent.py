import numpy as np
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

import matplotlib.pyplot as plt

##
# Data set simple
##

zer = np.zeros(5)
uno = np.ones(5)
set_dat = np.concatenate((zer,uno,zer,zer,zer))
reset_dat = np.concatenate((zer,zer,zer,uno,zer))
output_dat = np.concatenate((zer,uno,uno,zer,zer))

# Plot triple para visualizar
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(set_dat, '-o', color = 'green', label = 'set')
ax2.plot(reset_dat, '-o', color = 'blue', label = 'reset')
ax3.plot(output_dat, '-o', color = 'red', label = 'output')
ax1.legend(numpoints = 1)
ax2.legend(numpoints = 1)
ax3.legend(numpoints = 1)
plt.ylim([-0.1,1.1])
plt.title('Batch simple')
plt.show()

##
# Parametros de la red
# dim_inp es el tamano de inputs de la red
# dim_rnn es el de las capas internas de la red recurrente
# dim_out el tamano de la salida
##

dim_inp = 2
dim_rnn = 50
dim_out = 1

##
# stateful: Boolean (default False). 
# If True, the last state for each sample at index i in a batch 
# will be used as initial state for the sample of index i in the 
# following batch.
##

model = Sequential()
model.add(SimpleRNN(input_dim = 2,
                    units = 50,
                    stateful = False,
                   # batch_input_shape = (25,10,2),
                    return_sequences = True))

# Que activacion usar ??
model.add(Dense(units = 1, activation = "sigmoid"))


##
# Formato de input de la red. CUIDADO, las redes neuronales recursivas se manejan con
# "timesteps", con lo cual la data es separada en batches
# https://stackoverflow.com/questions/38294046/simple-recurrent-neural-network-input-shape
##


# Shape (2,data_length)
sample_size = 1000
in_list = np.array([set_dat, reset_dat])
print(np.shape(in_list))
x_train = np.array([[in_list] * sample_size]).reshape(len(set_dat),sample_size,2)
print(np.shape(x_train))
y_train = np.array([[output_dat] * sample_size]).reshape(len(output_dat),sample_size,1)
print(np.shape(y_train))

# Paso de compilacion
# For a mean squared error regression problem
#model.compile(optimizer='rmsprop',
#              loss='mse')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
history = model.fit(x_train, y_train,
                    epochs=100, batch_size = 100)

test_in =  np.array([[in_list] * 1]).reshape(len(set_dat),1,2)
out = np.array(model.predict(test_in))
out = out[:,0,0]
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
ax1.plot(output_dat, '-o', color = 'red', label = 'output real')
ax2.plot(out, '-o', color = 'blue', label = 'output red')
ax1.legend(numpoints = 1)
ax2.legend(numpoints = 1)
plt.ylim([-0.1,1.1])
plt.show()

