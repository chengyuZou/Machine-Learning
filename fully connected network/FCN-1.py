from random import shuffle
import paddle
import matplotlib.pyplot as plt
import os
import numpy as np
from paddle.nn.functional import mse_loss

#设置paddle默认的全局类型
paddle.set_default_dtype("float64")
#加载数据
train_dataset = paddle.text.datasets.UCIHousing(mode = 'train')
eval_dataset = paddle.text.datasets.UCIHousing(mode = 'test')

#封装训练数据
train_loader = paddle.io.DataLoader(train_dataset , batch_size = 32 , shuffle = True)
#封装验证数据
eval_loader = paddle.io.DataLoader(eval_dataset , batch_size= 8 ,shuffle=True)

#定义全连接网络
class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor , self).__init__()
        #定义一层全连接层,输出维度为1,激活函数为None
        self.linear = paddle.nn.Linear(13 , 1 ,None)

    #向前传播函数
    def forward(self , inputs):
        x = self.linear(inputs)
        return x

#记录批次
Batch = 0
Batchs = []
#记录损失值,用于后续绘图
all_train_loss = []

model = Regressor()  #模型实例化
model.train()        #训练模式
mse_loss = paddle.nn.MSELoss()  #均方误差损失函数
opt = paddle.optimizer.SGD(learning_rate= 0.0005 , parameters= model.parameters())  #随机梯度下降优化器
epochs_num = 200  #迭代次数

for pass_num in range(epochs_num):
    for batch_id , data in enumerate(train_loader()):
        image = data[0]
        label  = data[1]
        predict = model(image)  #向前计算
        loss = mse_loss(predict , label)

        if batch_id !=0 and batch_id %10 ==0:    #每10次记录一次并计算损失值
            Batch +=10
            Batchs.append(Batch)
            all_train_loss.append(loss.numpy().item())
            print("epoch:{}, step:{}, train_loss:{}".format(pass_num , batch_id , loss.numpy().item()))

            # epoch: 0, step: 10, train_loss: 882.5092158513136
            # epoch: 1, step: 10, train_loss: 708.1123880788004
            # epoch: 2, step: 10, train_loss: 630.5315794382199
            # epoch: 3, step: 10, train_loss: 599.8895382234077
            # epoch: 4, step: 10, train_loss: 752.3343602576886
            # epoch: 5, step: 10, train_loss: 608.9641095518352
            #......
            # epoch: 194, step: 10, train_loss: 44.978443222594535
            # epoch: 195, step: 10, train_loss: 88.87890573475251
            # epoch: 196, step: 10, train_loss: 35.7423576720849
            # epoch: 197, step: 10, train_loss: 53.611397694652965
            # epoch: 198, step: 10, train_loss: 133.17160368981166
            # epoch: 199, step: 10, train_loss: 96.5704198488326

        #反向传播
        loss.backward()
        opt.step()
        #重置梯度
        opt.clear_grad()

paddle.save(model.state_dict() , 'Regressor')  #保存模型


#进行绘图
def draw(Batchs , train_accs):
    title = "Training accs"
    plt.title(title , fontsize = 24)
    plt.xlabel("batch" , fontsize = 14)
    plt.ylabel("acc" , fontsize = 14)
    plt.plot(Batchs , train_accs , color = 'green' , label = 'training accs')
    plt.legend()
    plt.grid()
    plt.show()

draw(Batchs , all_train_loss)


#验证
para_state_dict = paddle.load("Regressor")  #加载模型参数
model = Regressor()                         #实例化模型
model.set_state_dict(para_state_dict)        #参数赋值
model.eval()            #验证模式

losses = []  # 用于存储每个批次的损失值
infer_results = []  # 用于存储所有的预测结果
ground_truths = []  # 用于存储所有的真实标签
for batch_id,data in enumerate(eval_loader()):
    image = data[0]
    label = data[1]
    ground_truths.extend(label.numpy())  # 转换为 NumPy 数组并加入列表
    predict = model(image)  # 使用模型进行前向推理，得到预测值
    infer_results.extend(predict.numpy())  # 转换为 NumPy 数组并加入列表
    loss = mse_loss(predict,label)
    losses.append(loss.numpy().item())
    avg_loss = np.mean(losses)
print("当前模型在验证集上的损失值为: " , avg_loss)
#当前模型在验证集上的损失值为:  17.713666204427053

#绘制真实值和预测值对比图
def draw_infer_result(ground_truths , infer_results):
    title = 'Boston'
    plt.title(title , fontsize = 24)
    x = np.arange(1, 20)
    y = x
    plt.plot(x ,y)
    plt.xlabel('ground truth' , fontsize =14)
    plt.ylabel('infer result' , fontsize = 14)
    plt.scatter(ground_truths , infer_results , color = 'green' , label = 'training cost')
    plt.grid()
    plt.show()

draw_infer_result(ground_truths , infer_results)


