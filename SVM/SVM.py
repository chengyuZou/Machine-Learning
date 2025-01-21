import numpy as np
from matplotlib import colors
from sklearn import svm
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl

#将字符串转化为整型
#该函数相当于映射 , Iris映射到0 ,1, 2
def iris_type(s):
    it = {b'Iris-setosa':0 , b'Iris-versicolor':1 , b'Iris-virginica':2}
    return it[s]

#加载数据
# converters 是一个字典，用于指定如何转换某些列的数据。
# 在这里，{4: iris_type} 表示将第 4 列（索引从0开始，第5列）应用 iris_type 转换函数
data = np.loadtxt('./iris.txt' ,
               dtype = float, #数据类型
               delimiter = ',',  #数据分隔符
               converters = {4 : iris_type}) #将标签用iris_type转换

#数据分割，将样本特征与样本标签分割
x , y = np.split(data , (4,) , axis = 1)
x = x[: , :2]  #取前两个特征分类
#调用函数进行训练集，测试集切分
x_train , x_test , y_train , y_test = model_selection.train_test_split(x , y , random_state=1 , test_size=0.2)

#SVM分类器构造
def classifier():
    clf = svm.SVC(C = 0.8 , #误差项惩罚系数
                 kernel='linear' ,
                  decision_function_shape= 'ovr') #决策函数
    return clf

#模型训练函数
def train(clf , x_train , y_train):
    clf.fit(x_train , y_train.ravel())  #训练集特征向量和 训练集目标值

#SVM模型定义
clf = classifier()

#调用函数训练模型
train(clf ,x_train , y_train )

#自定义准确率计算方法
def show_accuracy(a  ,b , tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy: %.3f' %(tip , np.mean(acc)))

#输出预测值
print('Training prediction: %.3f' %(clf.score(x_train , y_train)))
print('Test prediction: %.3f' %clf.score(x_test , y_test))

show_accuracy(clf.predict(x_train) , y_train , 'training_data')
show_accuracy(clf.predict(x_test) , y_test , 'testing_data')

#计算决策函数的值,表示x到各个分割平面的距离
#print('desicion_function: \n' , clf.decision_function(x_train)[:2])

# Training prediction: 0.808
# Test prediction: 0.767
# training_data Accuracy: 0.808
# testing_data Accuracy: 0.767
# desicion_function:
#  [[-0.24991711  1.2042151   2.19527349]
#  [-0.30144975  1.25525744  2.28694265]]

#绘图
def draw(clf, x):
    iris_feature = 'sepal length' , 'sepal_width' , 'petal length' , 'petal width'
    #获取第1 , 2维特征的最大值和最小值
    x1_min , x1_max = x[:, 0].min() , x[: , 0].max()
    x2_min , x2_max = x[: , 1].min() , x[: , 1].max()

    #生成网格采样点

    # 使用np.mgrid生成一个二维网格。200j表示在x1和x2范围内分别生成200个点。
    # 这个网格用于计算决策边界和分类区域。
    x1 , x2 = np.mgrid[x1_min : x1_max:200j , x2_min : x2_max :200j]
    #生成样本点
    # 将x1和x2生成的网格点扁平化，并堆叠成二维数组
    # grid_test，每行表示一个二维点的坐标。
    grid_test = np.stack((x1.flat , x2.flat) , axis=1)
    print('grid_test: \n' , grid_test[:2])
    # grid_test:
    # [[4.3       2.]
    #  [4.3       2.0120603]]

    #计算样本到决策面的距离
    z = clf.decision_function(grid_test)
    print('the distance to decision plane: \n' , z[:2])
    # the distance to decision plane:
    # [[1.15418548  2.24935988 - 0.26432263]
    #  [1.15805875  2.2485129 - 0.26434377]]

    grid_hat = clf.predict(grid_test)
    #预测分类值
    print('grid_hat: \n' , grid_hat[:2])

    # grid_hat:
    # [1. 1.]
    grid_hat = grid_hat.reshape(x1.shape)
    cm_light = mpl.colors.ListedColormap(['#A0FFA0' , '#FFA0A0' , '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g' , 'b' , 'r'])

    #绘制分类区域:能直观地表示分类边界
    plt.pcolormesh(x1 , x2 ,grid_hat , cmap = cm_light)
    #训练集与测试集数据:散点图
    # 使用scatter绘制训练数据点（x）和测试数据点（x_test）。
    # c参数用于设置颜色，s设置点的大小，cmap指定颜色映射
    plt.scatter(x[: , 0] , x[: ,1], c =np.squeeze(y) , edgecolors='k' , s  =50 ,cmap = cm_dark)
    plt.scatter(x_test[: ,0],x_test[: ,1] , s =120 , facecolor = 'none' , zorder = 10)

    plt.xlabel(iris_feature[0] , fontsize = 20)
    plt.ylabel(iris_feature[1] , fontsize = 20)
    plt.xlim(x1_min , x1_max)
    plt.ylim(x2_min , x2_max)
    plt.title('Iris data classification via SVM' , fontsize = 20)
    plt.grid()
    plt.show()

draw(clf , x)


