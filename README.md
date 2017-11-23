

Redes Neurais Artificiais com Tensorflow (Keras e TFLearn)
==========================================================

    
(Windows/Linux)
Instalação do Python 64-bits:
-------
É importante baixar a versão 64-bits, pois o TensorFlow não é compatível a versão 32 bits do Python.

https://www.python.org/ftp/python/3.5.2/python-3.5.2-amd64.exe (versão 3.5.2 64bits)

Para todas as versões:	
https://www.python.org/ftp/python/ 


 Instalando bibliotecas importantes (Numpy, Scipy, Scikit-learn e Pandas)
-------------

A instalação das bibliotecas será feita através do *pip*, o qual já vem por padrão no Python, através do CMD (Windows) ou Sheel (Linux).


 ## Numpy ##

    pip install numpy

 ## Scipy ##

    pip install scipy

 ## Scikit-learn ##

    pip install scikit-learn

 ## Pandas ##

    pip install pandas


Instalando Tensorflow
-------------
	
	pip install tensorflow

Instalando Keras e TFLearn
-------------
Keras e TFLearn são duas abstrações de alto nível do Tensorflow, que facilitam o a criação de modelos de redes. 	
		
		pip install keras
		pip install tflearn
		
Caso o *pip* não encontre automaticamente algum pacote, é possível baixá-lo manualmente através do repositório:
https://www.lfd.uci.edu/~gohlke/pythonlibs/

É necessário baixar o pacote de acordo com a respectiva versão do python instalada, além da arquitatura do mesmo. Por exemplo,
para instalação manual do numpy através do Python 3.6 -64bits:
-> Fazer download do arquivo *numpy‑1.13.3+mkl‑cp36‑cp36m‑win_amd64.whl*
-> instalar através do *pip* 
		pip install diretorio/numpy‑1.13.3+mkl‑cp36‑cp36m‑win_amd64.whl

-------------		
