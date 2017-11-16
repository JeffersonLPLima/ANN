Classificação Multi-classes com Keras
===================
Importando os módulos 
-------------

    import pandas
    from sklearn.cross_validation import train_test_split
	from sklearn.preprocessing import LabelEncoder
	from sklearn.cross_validation import train_test_split
	from sklearn.preprocessing import LabelEncoder
	from keras.models import Sequential
	from keras.layers.core import Dense, Activation
	from keras.utils import np_utils
	



 Carregando o dataset
-------------
Antes de executar essa linha, certifique-se de ter baixado a base de [dados](iris.csv).

	dataframe  = pandas.read_csv("iris.csv", header=0)
	dataset  = dataframe.values


Preparando os conjuntos de treino e teste
-------------
	
	X = dataset[:,0:4].astype(float)
	y = dataset[:,4]
	encoder = LabelEncoder() #Codifica as classes (String) em valores inteiros.
	encoder.fit(y)
	encoded_Y = encoder.transform(y)
	
	bin_y = np_utils.to_categorical(encoded_Y) #Binariza as classes (one hot encoding) 
						   #e.g Para um problema com três classes: 1 -> [1, 0, 0]   2-> [0, 1, 0]  3->  3->[0,0,1]
	X_train,X_test,y_train,y_test=train_test_split(X,bin_y,train_size=0.5,random_state=1)


Criando a rede
-------------

	model=Sequential()
	#A camada Dense funciona como "ativacao(dot(input, pesos) + bias)" onde a função de ativação é passada como argumento.
	                #16 = dimensão da saída
	model.add(Dense(16,input_shape=(4,))) #input_shape=(n_features,))) 
	model.add(Activation("sigmoid"))
	
	model.add(Dense(3))  #camada de saída obrigatoriamente tem que ser igual à dimensão dos labels		     
	model.add(Activation("softmax")) 
	
	model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
	
	
	model.fit(X_train,y_train,nb_epoch=3,batch_size=1,verbose=1)

Avaliando o desempenho da rede criada
-----------------------

	loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
	print("Accuracy = {:.2f}".format(accuracy))

 
