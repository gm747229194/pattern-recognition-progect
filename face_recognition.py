import cv2
import numpy as np
import os
from sklearn import neighbors
import tkinter
from tkinter import filedialog
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as knn
import matplotlib.pyplot as plt

#读取人脸数据库
#准备训练数据

def loadimages(data):
    
    '''
    data:train content
    images:[m,height,width]
    m样本数
    height高
    width宽
    name:名字的集合
    label：标签
    '''
    
    images = []
    labels = []
    names =[]
    label=1
    #读取照片所在的所有文件夹
    for subdirname in os.listdir(data):
        subjectpath = os.path.join(data,subdirname)
        if os.path.isdir(subjectpath):
            
            #一个文件夹一个人照片
            names.append(subdirname)
            for filename in os.listdir(subjectpath):
                imgpath = os.path.join(subjectpath,filename)
                img = cv2.imread(imgpath,0)  #OR cv2.IMREAD_GRAYSCALE
                images.append(img)
                labels.append(label)
            label = label+1
    images = np.asarray(images) #将列表转到数组，一张图片对应数组一行
    labels =np.asarray(labels)
    return images,labels,names

class face(object):
    def __init__(self,dsize=(46,56)):
        '''
        dimnum:pca降维后的维度
        neighbor：knn参数
        dsize：图像预处理的尺寸
        '''
        self._dsize = dsize

    def _prepare(self,images):
        '''
        图片的预处理，直方图均衡化
        images：训练集数据，灰度图片
        [m,height,width] m样本数 height高width宽
        return 处理后的数据[m,n]
        特征数n=dsize[0]*dsize[1]
        '''
        new_images = []
        for image in images:
            re_img = cv2.resize(image,self._dsize)
        #直方图均衡化
            hist_img = cv2.equalizeHist(re_img)
        #转换为一行
            hist_img = np.reshape(hist_img,(1,-1))
            new_images.append(hist_img)
        new_images = np.asarray(new_images)#列表变为数组
        return np.squeeze(new_images)

if __name__=='__main__':
    Face = face()
    #导入训练集训练
    data_train = "C:\\Users\\gm176\\Desktop\\project\\PCA-KNN\\image datebase_new\\train set"
    x_train,y_train,names = loadimages(data_train)
    x_trainNEW = np.squeeze(Face._prepare(x_train))

    pca = PCA(n_components=200).fit(x_trainNEW)
    x_trainreduced = pca.transform(x_trainNEW)
    clf = knn(n_jobs=-1,n_neighbors=1,weights='uniform')
    clf.fit(x_trainreduced, y_train)

    #导入测试集验证准确程度
    data_test = "C:\\Users\\gm176\\Desktop\\project\\PCA-KNN\\image datebase_new\\test set"
    x_test,y_test,names = loadimages(data_test)
    x_testNEW = np.squeeze(Face._prepare(x_test))
    x_testreduced = pca.transform(x_testNEW)
    #print(clf.score(x_trainreduced, y_train))
    print("PCA+KNN................")
    print(clf.score(x_testreduced, y_test))

    #compute and show average face
    train = np.genfromtxt('train.csv', delimiter=',').astype(int)
    x_train=train[:,:19200]
    x_trainAverage = np.mean(x_train, axis=0)
    x_trainAverage = x_trainAverage.reshape(120,160)
    plt.imshow(x_trainAverage, cmap='gray', vmin=0, vmax=255)
    plt.show()