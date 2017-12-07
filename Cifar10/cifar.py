from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


x_val = x_train[50000:60000]
y_val = y_train[50000:60000]



x_train = x_train[0:50000]
y_train = y_train[0:50000]


 
y_train = keras.utils.to_categorical(y_train, num_classes)  #convertendo as classes em vetores binários (one hot encoding)
y_test = keras.utils.to_categorical(y_test, num_classes)  
y_val = keras.utils.to_categorical(y_val, num_classes)  




model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


	#**Reduz a taxa de aprendizado**
	reduce_lr = ReduceLROnPlateau(monitor='val_loss',  #observar a função de perda no conjunto de validação
								factor=0.2,     #fator pela qual a taxa será reduzida -> nova_taxa = taxa_atual * factor
								verbose=1, 
								patience=1,    # número de épocas, sem redução na função de perda, que o monitor espera para agir.
								min_lr=0.0001) 	#limite inferior da nova_taxa. Menor que isso e o monitor não irá reduzir mais a taxa.

	#**Interrompe o treinamento**
	early_stopping=EarlyStopping(monitor='val_loss', #observar a função de perda no conjunto de validação
								patience=4,    # número de épocas, sem redução na função de perda, que o monitor espera para agir.
								verbose=1,)    
												

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 32
num_classes = 10
epochs = 100 
 #fazendo uso dos monitores criados		
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, early_stopping], verbose=1, validation_data=(x_val, y_val)) 
 
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
