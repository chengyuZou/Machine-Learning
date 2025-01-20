import numpy as np
import os,struct
import matplotlib.pyplot as plt
from array import array  as pyarray
from numpy import append , array , int8 , uint8 , zeros
from sklearn.metrics import accuracy_score, classification_report


def load_mnist(image_file , label_file , path = "D:/Ml-databases/MNIST"):
    digits = np.arange(10)

    fname_image = os.path.join(path , image_file)
    fname_label = os.path.join(path , label_file)

    # flbl = open(fname_label, 'rb')：以二进制模式打开标签文件
    # fname_label。

    # 其中
    # fname_label
    # 是标签文件的完整路径，例如
    # train - labels - idx1 - ubyte。
    # magic_nr, size = struct.unpack(">II", flbl.read(8))：
    #
    # 读取文件的前8个字节。
    # struct.unpack(">II", ...)
    # 将这8个字节解压成两个整数：
    # magic_nr
    # 是文件格式的魔术数（通常是用于验证文件格式的标识符）。
    # size
    # 是标签的总数量，即数据集中图像对应的标签数量。
    # lbl = pyarray("b", flbl.read())：
    #
    # 使用
    # flbl.read()
    # 读取剩余的标签数据。
    # pyarray("b", ...)
    # 将读取的字节数据转换为
    # Python
    # 数组，其中
    # b
    # 表示“字节”类型（一个标签是一个字节，表示一个数字标签）。所以，lbl
    # 是包含所有标签的数组。
    # flbl.close()：关闭标签文件。

    flbl = open(fname_label , 'rb')
    magic_nr  , size = struct.unpack(">II" , flbl.read(8))
    lbl = pyarray("b" , flbl.read())
    flbl.close()

    # fimg = open(fname_image, 'rb')：以二进制模式打开图像文件
    # fname_image。
    #
    # 其中
    # fname_image
    # 是图像文件的完整路径，例如
    # train - images - idx3 - ubyte。
    # magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))：
    #
    # 读取文件的前16个字节。
    # struct.unpack(">IIII", ...)
    # 将这16个字节解压成四个整数：
    # magic_nr
    # 是文件格式的魔术数（同样用于文件验证）。
    # size
    # 是图像的数量，通常和标签文件中的标签数量相同。
    # rows
    # 和
    # cols
    # 是图像的尺寸（MNIST
    # 数据集中的图像是28x28，即
    # rows = 28，cols = 28）。
    # img = pyarray("B", fimg.read())：
    #
    # 使用
    # fimg.read()
    # 读取剩余的图像数据。
    # pyarray("B", ...)
    # 将这些字节数据转换为
    # Python
    # 数组，其中
    # B
    # 表示“无符号字节”类型（每个图像是一个28x28的像素矩阵，每个像素是一个字节）。所以，img
    # 是一个包含所有图像数据的数组。
    # fimg.close()：关闭图像文件。

    fimg = open(fname_image , 'rb')
    magic_nr , size , rows , cols = struct.unpack(">IIII" , fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = np.zeros((N, rows*cols) ,dtype=uint8 )
    labels = np.zeros((N,1) , dtype=int8)

    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i]* rows * cols : (ind[i]+1) * rows * cols]).reshape((1, rows*cols))
        labels[i] = lbl[ind[i]]

    return images , labels

# images, labels = load_mnist(image_file, label_file, path)
#
# # 查看图像和标签的形状
# print(images.shape)  # 应该输出 (N, 784)，其中 N 是图像数量
# print(labels.shape)  # 应该输出 (N, 1)，其中 N 是标签数量

#matlab显示图片,可不用
def show_image(imgdata , imgtarget , show_colunm , show_row):
    for index ,(im ,it) in enumerate (list(zip(imgdata , imgtarget))):
        xx = im.reshape(28 , 28)
        plt.subplots_adjust(wspace=1, hspace=2.5)
        plt.subplot(show_row , show_colunm , index +1)
        plt.axis('off')
        plt.imshow(xx , cmap = 'gray' , interpolation='nearest')
        plt.title('%i' % it)

    plt.show()

train_image , train_label = load_mnist('train-images.idx3-ubyte' , 'train-labels.idx1-ubyte')
test_image , test_label = load_mnist('t10k-images.idx3-ubyte' , 't10k-labels.idx1-ubyte')

#show_image(train_image[: 50] , train_label[:50] , 10 ,5)

#导入类
from sklearn.linear_model import LogisticRegression
#实例化类
lr = LogisticRegression()

#数据预处理

#缩放到0~1区间
train_image = [im / 255.0 for im in train_image]
test_image = [im / 255.0 for im in test_image]
#训练模型
lr.fit(train_image , train_label)

predict = lr.predict(test_image)

#打印准确率 分类指标
print("accuracy_score: %.4lf" % accuracy_score(predict , test_label))
print("Classificatioon report for classifier %s : \n %s \n" % (lr , classification_report(test_label , predict)))


#输出
# accuracy_score: 0.9258
# Classificatioon report for classifier LogisticRegression() :
#                precision    recall  f1-score   support
#
#            0       0.95      0.98      0.96       980
#            1       0.96      0.98      0.97      1135
#            2       0.93      0.90      0.91      1032
#            3       0.90      0.91      0.91      1010
#            4       0.94      0.93      0.93       982
#            5       0.91      0.88      0.89       892
#            6       0.94      0.95      0.94       958
#            7       0.94      0.92      0.93      1028
#            8       0.87      0.88      0.88       974
#            9       0.91      0.92      0.92      1009
#
#     accuracy                           0.93     10000
#    macro avg       0.92      0.92      0.92     10000
# weighted avg       0.93      0.93      0.93     10000





