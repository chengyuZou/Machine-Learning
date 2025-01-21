import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import  datasets

#直接导入数据集
iris = datasets.load_iris()
X = iris.data[: , :4]  #提取特征空间的4个维度

#欧氏距离定义
def distEclud(x , y):
    return np.sqrt(np.sum((x - y) **2))

#定义簇心,此处使用随机抽取的k个样本点为簇心进行后续计算
def randCent(dataSet , k):
    m , n = dataSet.shape #m  =150 , n=4
    centroids = np.zeros((k,n)) #k*4
    for i in range(k):
        index = int(np.random.uniform(0  ,m)) #产生0到150的随机数
        centroids[i,:] = dataSet[index , :]
    return centroids

#k均值聚类算法
def KMeans(dataSet , k):
    m = np.shape(dataSet)[0]  #样本数
    # np.mat() 创建150 * 2的矩阵
    #第一列存每个样本属于哪一簇, 第二列存每个样本到簇心的误差
    clusterAssment = np.mat(np.zeros((m ,2)))
    clusterChange = True

    #初始化质心
    centroids = randCent(dataSet , k)
    while clusterChange:
        #样本所属簇不再变化停止迭代
        clusterChange = False

        #遍历所有样本
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            #遍历所有的簇心
            for j in range(k):
                #找到该样本到k个簇心的欧式距离
                #找到距离最近的那个簇心minIndex
                distance = distEclud(centroids[j , :] , dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex  =j

            # 更新该行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2

        #更新簇心
        for j in range(k):
            # 获取对应簇类所有的点
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            # 求均值,产生新的质心
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    return centroids , clusterAssment


def draw(data , center , assment):
    length = len(center)
    fig = plt.figure
    data1 = data[np.nonzero(assment[: ,0].A == 0)[0]]
    data2 = data[np.nonzero(assment[:, 0].A == 1)[0]]
    data3 = data[np.nonzero(assment[:, 0].A == 2)[0]]
    #选择前两个维度绘制原始数据的散点图
    plt.scatter(data1[: ,0] , data1[:,1],  c='red' , marker = 'o' , label = 'label0')
    plt.scatter(data2[:, 0], data2[:, 1], c='green', marker='*', label='label1')
    plt.scatter(data3[:, 0], data3[:, 1], c='blue', marker='+', label='label2')

    #绘制簇的质心点
    for i in range(length):
        plt.annotate('center' , xy = (center[i,0] , center[i,1]) , xytext = \
            (center[i , 0]+1 , center[i,1]+1) , arrowprops=dict(facecolor = 'yellow'))
    plt.show()

    #选取后两个维度绘制原始数据的散点图
    plt.scatter(data1[: ,2] , data1[:,3],  c='red' , marker = 'o' , label = 'label0')
    plt.scatter(data2[:, 2], data2[:, 3], c='green', marker='*', label='label1')
    plt.scatter(data3[:, 2], data3[:, 3], c='blue', marker='+', label='label2')

    #绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 2], center[i, 3]), xytext= \
            (center[i, 2] + 1, center[i, 3] + 1), arrowprops=dict(facecolor='yellow'))
    plt.show()


#执行算法
dataSet = X
k = 3
centroids , clusterAssment = KMeans(dataSet, k)
draw(dataSet , centroids , clusterAssment)

