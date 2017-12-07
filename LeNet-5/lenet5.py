from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


#LeNet
#
#CONV = RELU = SUBSAMPLING -> CONV = RELU = SUBSAMPLING -> FLATTEN = FULLY CONNECTED = RELU -> FULLY CONNECTED = SOFTMAX -> OUTPUT (n_classes)
#
#
model = Sequential()

model.add(Convolution2D(20, 5, 5, border_mode="same",
    input_shape=(depth, height, width)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# softmax classifier
model.add(Dense(classes))
model.add(Activation("softmax"))


 