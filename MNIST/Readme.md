Redes Neurais Convolucionais na base MNIST
===================
Importando os módulos 
-------------

    import keras
	from keras.datasets import mnist
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
	from keras import backend as K
	from keras.callbacks import ReduceLROnPlateau, EarlyStopping
		



 Carregando o dataset
-------------
		 
	#dimensões das imagens
	img_rows, img_cols = 28, 28

	#Na primeira execução baixa o dataset, na segunda só carrega
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#ajustando o tamanho da matriz de treino
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
	
	#ajustando o tamanho da matriz de teste
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	
	#definindo o tamanho da entrada da rede 
	input_shape = (img_rows, img_cols, 1)  
		


Preparando os conjuntos de treino e teste
-------------
	


	#transformando os píxels das imagens em floats
	x_train = x_train.astype('float32') 
	
	#transformando os píxels das imagens em floats 
	x_test = x_test.astype('float32')
	x_train /= 255    #fixando intervalo [0, 1]
	x_test /= 255   #fixando intervalo [0, 1]

	#Separando o conjunto de validação. 10 mil amostras para validação e 50 mil para treino.
	x_val = x_train[50000:60000]
	y_val = y_train[50000:60000]
	
	x_train = x_train[0:50000]
	y_train = y_train[0:50000]
	
	 #convertendo as classes em vetores binários (one hot encoding) https://www.quora.com/What-is-one-hot-encoding-and-when-is-it-used-in-data-science
	 
	y_train = keras.utils.to_categorical(y_train, num_classes) 
	y_test = keras.utils.to_categorical(y_test, num_classes)  
	y_val = keras.utils.to_categorical(y_val, num_classes)
	
    

Criando a rede
-------------

	model = Sequential()      #Criando o modelo
	
	#Camada de convolução 2d, com janela 3x3, e Adcionando a ativação relu.
	model.add(Conv2D(64, (3, 3), activation='relu')) 
	
	#Camada de Pooling 2d, de tamanho 2x2, através da média. (Retira a média aritmética de janelas 2x2 na imagem, reduzindo a quantidade de características 4->1 por janela)  
	model.add(AveragePooling2D(pool_size=(2, 2)))
	
	 #Camada de Dropout para evitar overfitting. Nada mais é do que desativar um neurônio. No caso a probabilidade de que isso seja feita é de 0.25, no exemplo em questão.         
	model.add(Dropout(0.25))   
	   
	#Reduz a dimensão da camada. output_shape == (None, 64, 32, 32) -> Flatten() -> output_shape == (None, 65536)
	model.add(Flatten())    
	
	#Camada completamente conectada														
	model.add(Dense(128, activation='relu'))  
	
	#Camada de saída, com ativação softmax.  
	model.add(Dense(num_classes, activation='softmax'))   

	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  
# monitores para o treinamento da rede
	#	https://keras.io/callbacks/
	

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
																


Avaliando o desempenho da rede criada
-----------------------

																	  #fazendo uso dos monitores criados		
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[reduce_lr, early_stopping], verbose=1, validation_data=(x_val, y_val))
	
	loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
	
	print('Test accuracy:', accuracy)
