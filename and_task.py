#And task with keras (Toy Task)

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# the four different states of the And gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[0],[0],[1]], "float32")

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint
filepath       ="weights-improvement-{epoch:02d}-{binary_accuracy:.2f}.hdf5"
checkpoint     = ModelCheckpoint(filepath, monitor='binary_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history     = model.fit(training_data, target_data, nb_epoch=500,callbacks=callbacks_list, verbose=2)
pepe        = model.save
nombres     = model.metrics_names
pesos       = model.get_weights()

print "training_data\n ",training_data 
print "target_data\n ",target_data
print "model prediction\n ", model.predict(training_data).round()
print "history keys",(history.history.keys())
print "---------------------------------"
print "model.get_weights()\n",pesos
print "model.get_weights()\n",len(pesos)
print"pesos 0\n",pesos[0]
print"pesos 1\n",pesos[1]
print"pesos 2\n",pesos[2]
print"pesos 3\n",pesos[3]
print"model.summary()\n",model.summary(),"\n"
print "---------------------------------"

scores = model.evaluate(training_data, target_data, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


x_pred = training_data[0:3]
y_pred = model.predict(training_data)
 
print"x_pred\n",x_pred,"y_pred\n",y_pred   
#plt.scatter(x_pred)
#plt.scatter(x_pred)
#plt.scatter(y_pred)
#plt.show()

#  "Accuracy"
plt.plot(history.history['binary_accuracy'])
#plt.plot(history.history['val_acc'])
plt.title('model binary accuracy')
plt.ylabel('binary accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
