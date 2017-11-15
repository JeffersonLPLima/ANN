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
Antes de executar essa linha, certifique-se de ter baixado a base de [dados].(https://github.com/JeffersonLPLima/ANN/blob/master/Iris%20Flower%20Classification%20%28Keras%29/iris.csv)

	dataframe  = pandas.read_csv("iris.csv", header=0)
	dataset  = dataframe.values


Preparando os conjuntos de treino e teste
-------------
	
	X = dataset[:,0:4].astype(float)
	y = dataset[:,4]
	encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(y)
	encoded_Y = encoder.transform(y)
	bin_y = np_utils.to_categorical(encoded_Y)
	X_train,X_test,y_train,y_test=train_test_split(X,bin_y,train_size=0.5,random_state=1)


Criando a rede
-------------

	model=Sequential()
	
	model.add(Dense(16,input_shape=(4,)))
	model.add(Activation("sigmoid"))
	
	model.add(Dense(3))
	model.add(Activation("softmax"))
	
	model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
	
	model.fit(X_train,y_train,nb_epoch=3,batch_size=1,verbose=1)

Avaliando o desempenho da rede criada
-----------------------

	loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
	print("Accuracy = {:.2f}".format(accuracy))

 
