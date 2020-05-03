from keras.datasets import cifar10
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
import matplotlib.pyplot as plt


#load dataset
np.random.seed(10)
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()

# normalization 
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0

# one-hot coding
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
x_train = np.array(x_img_train_normalize)

model = Sequential()

# 1st convolutional layer，32 filters 3x3 ，Activation function relu
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(32, 32,3), 
                 activation='relu', 
                 padding='same'))

# Dropout 25% neurons
model.add(Dropout(0.25))

# Maxpooling layer，window 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd convolutional layer，64 filters 3x3 ，Activation function relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', padding='same'))

# Dropout 25% neurons
model.add(Dropout(0.25))

# Maxpooling layer，window 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())
model.add(Dropout(rate=0.25))
## Classification
# Dense layer
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))
model.summary()

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x_img_train_normalize, y_label_train_OneHot,
                          validation_split=0.2,
                          epochs=10, batch_size=128, verbose=1) 
scores = model.evaluate(x_img_test_normalize,
                        y_label_test_OneHot, verbose=0)

#输出结果
print('training accuracy--------')
print(scores[0])
print('testing accuracy--------')
print(scores[1])

