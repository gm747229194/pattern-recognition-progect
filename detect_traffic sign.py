import sys, os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
from keras.utils  import np_utils
sys.path.append('C:\\Users\\gm176\\Desktop\\traffic-data')

# 数据集所在的文件位置
training_file = "C:\\Users\\gm176\\Desktop\\traffic-data\\train.p"
validation_file = "C:\\Users\\gm176\\Desktop\\traffic-data\\valid.p"
testing_file = "C:\\Users\\gm176\\Desktop\\traffic-data\\test.p"

# 打开文件
with open(training_file,mode="rb") as f:
    train = pickle.load(f)
with open(validation_file,mode="rb") as f:
    valid = pickle.load(f)
with open(testing_file,mode="rb") as f:
    test = pickle.load(f)

# 获取数据集的特征及标签数据
X_train,y_train = train["features"],train["labels"]
X_valid,y_valid = valid["features"],valid["labels"]
X_test,y_test = test["features"],test["labels"]

print("Number of training examples =",X_train.shape[0])
print("Number of validtion examples =",X_valid.shape[0])
print("Number of testing examples=",X_test.shape[0])

# 查看数据格式
print("Image data shape =",X_train.shape[1:])

# 查看数据的标签的数量
sum = np.unique(y_train)
print("number of classes =",len(sum))

# 查看标签数据
sign_names_file = "C:\\Users\\gm176\\Desktop\\traffic-data\\signnames.csv"
sign_names = pd.read_csv(sign_names_file)
#print(sign_names)

# 定义将标签id转换成name的函数
sign_names = np.array(sign_names)
def id_to_name(id):
    return sign_names[id][1]

def preprocess_features(X, equalize_hist=True):
    normalized_X = []
    for i in range(len(X)):
        # Convert from RGB to YUV
        yuv_img = cv2.cvtColor(X[i], cv2.COLOR_RGB2YUV)
        yuv_img_v = X[i][:, :, 0]
        # equalizeHist
        yuv_img_v = cv2.equalizeHist(yuv_img_v)
        # expand_dis
        yuv_img_v = np.expand_dims(yuv_img_v, 2)
        normalized_X.append(yuv_img_v)
    # normalize
    normalized_X = np.array(normalized_X, dtype=np.float32)
    normalized_X = (normalized_X-128)/128
    # normalized_X /= (np.std(normalized_X, axis=0) + np.finfo('float32').eps)
    return normalized_X

# 对数据集整体进行处理
X_train_normalized = preprocess_features(X_train)
X_valid_normalized = preprocess_features(X_valid)
X_test_normalized = preprocess_features(X_test)

# 将数据集打乱
X_train_normalized,y_train = shuffle(X_train_normalized,y_train)

# 数据增强
'''
from keras.preprocessing.image import ImageDataGenerator

# 图像数据生成器
image_datagen = ImageDataGenerator(rotation_range = 10.,
                                   zoom_range = 0.2,
                                   width_shift_range =  0.08,
                                   height_shift_range = 0.08
                                  )

# 从训练集随意选取一张图片
index = np.random.randint(0, len(X_train_normalized))
img = X_train_normalized[index]

# 展示原始图片
plt.figure(figsize=(1, 1))
plt.imshow(np.squeeze(img), cmap="gray")
plt.title('Example of GRAY image (name = {})'.format(id_to_name(y_train[index])))
plt.axis('off')
plt.show()

# 展示数据增强生成的图片
fig, ax_array = plt.subplots(3, 10, figsize=(15, 5))
for ax in ax_array.ravel():
    images = np.expand_dims(img, 0)
    # np.expand_dims(img, 0) means add dim
    augmented_img, _ = image_datagen.flow(np.expand_dims(img, 0), np.expand_dims(y_train[index], 0)).next()
    #augmented_img=preprocess_features(augmented_img)
    ax.imshow(augmented_img.squeeze(), cmap="gray")
    ax.axis('off')
plt.suptitle('Random examples of data augment (starting from the previous image)')
plt.show()
'''

# 对标签数据进行one-hot编码

n_classes = len(sum)
#print("Shape before one-hot encoding:",y_train.shape)
Y_train = np_utils.to_categorical(y_train,n_classes)
Y_valid = np_utils.to_categorical(y_valid,n_classes)
Y_test = np_utils.to_categorical(y_test,n_classes)
#print("Shape after one-hot encoding:",Y_train.shape)
#print(y_train[0])
#print(Y_train[0])
print(X_train_normalized.shape)
print(Y_train.shape)

# 读入数据
x_train, t_train = X_train_normalized, Y_train
x_test, t_test = X_test_normalized, Y_test
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
## Feature Extraction
# 第1层卷积，32个3x3的卷积核 ，激活函数使用 relu
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=X_train_normalized.shape[1:]))

# 第2层卷积，64个3x3的卷积核，激活函数使用 relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# 最大池化层，池化窗口 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout 25% 的输入神经元
model.add(Dropout(0.25))

# 将 Pooled feature map 摊平后输入全连接网络
model.add(Flatten())

## Classification
# 全联接层
model.add(Dense(128, activation='relu'))

# Dropout 50% 的输入神经元
model.add(Dropout(0.5))

# 使用 softmax 激活函数做多分类，输出各类别的概率
model.add(Dense(n_classes, activation='softmax'))
model.summary()

# 编译模型
model.compile(loss="categorical_crossentropy",
              metrics=["accuracy"],
              optimizer="adam")

# 训练模型
history = model.fit(X_train_normalized,
                    Y_train,
                    batch_size=256,
                    epochs=30,
                    verbose=2,
                    validation_data=(X_valid_normalized,Y_valid))     
scores = model.evaluate(X_valid_normalized,
                        Y_valid, verbose=0)

#输出结果
print('testing loss--------')
print(scores[0])
print('testing accuracy--------')
print(scores[1])