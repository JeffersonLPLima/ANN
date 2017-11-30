from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


batch_size = 128
num_classes = 10
epochs = 20

img_rows, img_cols = 28, 28 #dimensões das imagens
(x_train, y_train), (x_test, y_test) = mnist.load_data() #Na primeira execução baixa o dataset, na segunda só carrega
x_train = x_train.reshape(60000, 784) #ajustando o tamanho da matriz de treino
x_test = x_test.reshape(10000, 784) #ajustando o tamanho da matriz de teste
 
x_train = x_train.astype('float32')  #transformando os píxels das imagens em floats
x_test = x_test.astype('float32') #transformando os píxels das imagens em floats
x_train /= 255    #fixando intervalo [0, 1]
x_test /= 255   #fixando intervalo [0, 1]

x_val = x_train[50000:60000]
y_val = y_train[50000:60000]



x_train = x_train[0:50000]
y_train = y_train[0:50000]


print(y_train[0:100])

y_train = keras.utils.to_categorical(y_train, num_classes)  #convertendo as classes em vetores binários (one hot encoding)
y_test = keras.utils.to_categorical(y_test, num_classes)  
y_val = keras.utils.to_categorical(y_val, num_classes)  


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

  
# monitores para o treinamento da rede
#https://keras.io/callbacks/

#Reduz a taxa de aprendizado
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  #observar a função de perda no conjunto de validação
							factor=0.2,     #fator pela qual a taxa será reduzida -> nova_taxa = taxa_atual * factor
							verbose=1, 
							patience=1,    # número de épocas, sem redução na função de perda, que o monitor espera para agir.
							min_lr=0.0001) 	#limite inferior da nova_taxa. Menor que isso e o monitor não irá reduzir mais a taxa.

#Interrompe o treinamento
early_stopping=EarlyStopping(monitor='val_loss', #observar a função de perda no conjunto de validação
							patience=4,    # número de épocas, sem redução na função de perda, que o monitor espera para agir.
							verbose=1,)    
																		
																  #fazendo uso dos monitores criados		
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, early_stopping], verbose=1, validation_data=(x_val, y_val))

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print('Test accuracy:', accuracy)
 
