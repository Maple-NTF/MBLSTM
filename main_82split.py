import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from torch.nn import functional as F

replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced.csv')
replaced_data_test = pd.read_csv('testdata_0.2_171.csv')#180

# 将class列的数据替换为1和0
replaced_data_train['class'] = replaced_data_train['class'].replace({'IntervalPeriod': 1, 'AcutePhase': 0})

#类型统一
traindata = replaced_data_train.astype('float64')
testdata = replaced_data_test.astype('float64')
print(traindata.head())
print(testdata.head())

# 提取特征和标签
X = traindata.iloc[:, :-1].values
Y = traindata.iloc[:, -1].values

# 使用SMOTE进行数据增强
# smote = SMOTE()
# X_smoted, Y_smoted = smote.fit_resample(X, Y)

#no smote
augmented_data = pd.DataFrame(X, columns=replaced_data_train.columns[:-1])
augmented_data['label'] = Y

# 将增强后的数据保存到新的CSV文件
# augmented_data = pd.DataFrame(X_smoted, columns=replaced_data_train.columns[:-1])
# augmented_data['label'] = Y_smoted
augmented_data.to_csv('augmented_data.csv', index=False)
print(augmented_data.shape)
print(augmented_data.head())
#到这步就是完成了数据增强的全部步骤，但是仍然没完成转换成模型能读取的数据格式的步骤

#将数据转换成模型可以读取的形式
augmented_data = augmented_data.astype('float64')
X_train = augmented_data.iloc[:, :-1].values
print(X_train[:5])
Y_train = augmented_data.iloc[:, -1].values
print(Y_train[:5])
X_train = X_train.astype('float64')
#为了解决Y_train的问题 创建LabelEncoder对象
label_encoder = LabelEncoder()
# 对Y_train进行编码
Y_train_encoded = label_encoder.fit_transform(Y_train)
# 将编码后的Y_train转换为float类型
Y_train = Y_train_encoded.astype('float64')

X_val = testdata.iloc[:, :-1].values
Y_val = testdata.iloc[:, -1].values
print(X_val[:5])
print(Y_val[:5])
X_val = X_val.astype('float64')
Y_val_encoded = label_encoder.fit_transform(Y_val)
Y_val = Y_val_encoded.astype('float64')

# 将输入数据转换为3D张量
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

# 定义交互模块
class InteractionModule(nn.Module):
    def __init__(self, units, num_iterations):
        super(InteractionModule, self).__init__()
        self.units = units
        self.num_iterations = num_iterations
        self.dense1 = nn.Linear(units, units)
        self.dense2 = nn.Linear(units, units)
        self.dense3 = nn.Linear(units, units)
        self.dense4 = nn.Linear(units, units)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, forward_hidden_state, backward_hidden_state):
        x0 = inputs
        # print(x0)
        hf = forward_hidden_state
        # print(hf)
        hb = backward_hidden_state
        # print(hb)

        for _ in range(self.num_iterations):
            # y1 = torch.cat([x0, hf], dim=-1)
            # print(y1.shape)
            # print(self.dense1)
            x1 = self.sigmoid(self.dense1(x0 + hf))
            # print(x1.shape)
            # print("************************")
            hb2 = self.sigmoid(self.dense2(hb + x1))
            # print(hb2.shape)
            # print("************************")
            hf2 = self.sigmoid(self.dense3(x1 + hf))
            # print(hf2.shape)
            # print("************************")
            x2 = self.sigmoid(self.dense4(hb2 + x1))
            # print(x2.shape)
            # print("￥￥￥￥￥￥￥￥￥￥￥￥￥￥")
            x0 = x2
            hf = hf2
            hb = hb2

            # print(x0)
            # print(hf)
            # print(hb)
        return x0, hf, hb

# 定义双向LSTM层
# class BidirectionalLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_iterations):
#         super(BidirectionalLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_iterations = num_iterations
#         self.forward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
#         self.backward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
#         # self.lin = nn.Linear(hidden_size * 1, hidden_size * 1)
#         # torch.nn.init.zeros_(self.lin.bias)
#         # torch.nn.init.xavier_uniform_(self.lin.weight)
#         print("input_size（after LSTM）:" ,input_size)
#         print("hidden_size（after LSTM）:", hidden_size)
#         self.interaction_output_forward_h = torch.zeros(1)
#         self.interaction_output_backward_h = torch.zeros(1)
#         self.interaction_module = InteractionModule(hidden_size, num_iterations)
#
#     def forward(self, inputs):
#         forward_output, _ = self.forward_lstm(inputs)
#         backward_output, _ = self.backward_lstm(inputs.flip(dims=[1]))
#         backward_output = backward_output.flip(dims=[1])
#
#         forward_h, forward_c = forward_output[:, :, :self.hidden_size], forward_output[:, :, self.hidden_size:]
#         backward_h, backward_c = backward_output[:, :, :self.hidden_size], backward_output[:, :, self.hidden_size:]
#
#         interaction_output, self.interaction_output_forward_h, self.interaction_output_backward_h = self.interaction_module(inputs, forward_h, backward_h)#获取了hf和hb
#         #interaction_output是定了是[2,1,15]，主要是为了能在各个交互模块之中的迭代中统一，主要原因是迭代时候维度为15，我希望能保持维度为15继续之后的迭代。
#         #因此现在要做的是在保持interaction_output已有的信息状态不变的情况下对output_h的内容进行处理。这部分的处理是将两个[2,1,15]的信息重新处理成一个新的[2,1,15]。
#         #最简单的办法就是求平均值，很无脑很离谱，但很高效，并且不容易出大错
#
#         # 现在已知interaction_output的tensor是[2,1,15]，interaction_output_forward_h和interaction_output_backward_h的tensor也是[2,1,15]。
#         # 我需要的是一个能够对后两者进行的操作，使其得到的output_h的tensor也依然是[2,1,15]从而可以和interaction_output继续进行下一步的操作
#         # 不用拼接的手段，使用不会改变维度的手段方便后续操作？我的目的是什么？仍然是需要两个输出的信息都可以保留
#         #不能用torch.cat、不能用torch.hstack、不能用torch.stack
#         output_h = (self.interaction_output_forward_h + self.interaction_output_backward_h) / 2
#
#
#         # print("output_h:",output_h)
#         # print("&&&&&&&&&&&&&&")
#         # print("interaction_output:", interaction_output)
#         # print("&&&&&&&&&&&&&&")
#         #需要处理两个输出，使其能转化为一个输出
#
#         combined_output = torch.cat([interaction_output, output_h], dim=1)
#         combined_output = torch.mean(combined_output, dim=1)
#         combined_output = self.lin(combined_output)
#         print("combined_output:", combined_output)
#         return combined_output

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_iterations, seq_len, batch_size):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.seq_len = seq_len
        # self.batch_size = batch_size

        # 定义前向LSTM单元
        self.forward_lstmcell = nn.LSTMCell(input_size, hidden_size)
        # self.forward_lstmcell = nn.LSTMCell(hidden_size, hidden_size)
        # 定义后向LSTM单元
        self.backward_lstmcell = nn.LSTMCell(input_size, hidden_size)
        # self.backward_lstmcell = nn.LSTMCell(hidden_size, hidden_size)

        # 定义交互模块
        self.interaction_output_forward_h = torch.zeros(1)
        self.interaction_output_backward_h = torch.zeros(1)
        self.interaction_module = InteractionModule(hidden_size, num_iterations)

    def forward(self, inputs):
        print("inputs:")
        print(inputs.shape)

        # 初始化前向LSTM隐藏状态和细胞状态
        self.interaction_output_forward_h= torch.zeros(seq_len, self.hidden_size)
        # forward_cell = torch.zeros(batch_size, self.hidden_size)
        forward_cell = torch.zeros(seq_len, self.hidden_size)

        # 初始化后向LSTM隐藏状态和细胞状态
        self.interaction_output_forward_h = torch.zeros(seq_len, self.hidden_size)
        # backward_cell = torch.zeros(batch_size, self.hidden_size)
        backward_cell = torch.zeros(seq_len, self.hidden_size)

        # 存储交互模块的输出
        # interaction_outputs = []

        # 前向传播
        for t in range(seq_len):
            forward_seq_now = inputs[t]
            backward_seq_now = inputs[seq_len - t - 1]

            # forward_hidden, forward_cell = self.forward_lstmcell(forward_seq_now, (forward_hidden, forward_cell))
            # backward_hidden, backward_cell = self.backward_lstmcell(backward_seq_now, (backward_hidden, backward_cell))

            # 交互模块
            interaction_output, self.interaction_output_forward_h, self.interaction_output_backward_h = self.interaction_module(
                inputs, self.interaction_output_forward_h, self.interaction_output_backward_h)  # 获取了hf和hb
            interaction_output = (self.interaction_output_forward_h + self.interaction_output_backward_h) / 2
            #interaction_outputs.append(interaction_output)
            # print(self.interaction_output_backward_h.shape)
            # print("464646")

            _, forward_cell[t] = self.forward_lstmcell(forward_seq_now, (self.interaction_output_forward_h[t], forward_cell[t]))
            _, backward_cell[seq_len - t - 1] = self.backward_lstmcell(backward_seq_now, (self.interaction_output_backward_h[seq_len - t - 1], backward_cell[seq_len - t - 1]))

        print("interaction_output:")
        print(interaction_output.shape)
        # 将交互模块的输出转换为张量
        return interaction_output

# 定义模型参数
input_size = X_train.shape[2]
print("input_size:" ,input_size)#15
hidden_size = 15#15#64
seq_len = 9
num_iterations = 2 #交互模块迭代两次
num_classes = 2
batch_size = 30#30
num_epochs = 400#300

# 定义模型
model = nn.Sequential(
    # BidirectionalLSTM(input_size, hidden_size, num_iterations),
    BiLSTM(input_size, hidden_size, num_iterations, seq_len, batch_size),
    nn.Linear(hidden_size * 1, hidden_size),#2
    nn.ReLU(),
    nn.Linear(hidden_size, num_classes),
    nn.Softmax(dim=-1)
)

def check(vec_a, vec_b):
    dot = 0
    for i in range(len(vec_a)):
        dot += vec_a[i] * vec_b[i]
    vec_a_sq_sum = math.sqrt(sum([item * item for item in vec_a]))
    vec_b_sq_sum = math.sqrt(sum([item * item for item in vec_b]))
    return dot / (vec_a_sq_sum * vec_b_sq_sum)


# 定义损失函数和优化器
#使用最大边缘损失加大正负样本之间的差距
criterion = nn.CrossEntropyLoss()#交叉熵损失

def contrasive_loss(outputs, label):
    x = outputs
    class_label = label
    #需要一句话分开正负样本的标签
    # pos_adj_mat = (class_label.unsqueeze(0) == class_label.unsqueeze(1)).byte()
    # neg_adj_mat = pos_adj_mat ^ 1
    # pos_adj_mat.fill_diagonal_(0)
    # pos_adj_mat = class_label[class_label == 1]
    # neg_adj_mat = class_label[class_label == 0]
    pos_adj_mat = (class_label.unsqueeze(0) == class_label.unsqueeze(1)).byte()
    neg_adj_mat = pos_adj_mat ^ 1
    pos_adj_mat.fill_diagonal_(0)
    pos = torch.tensor(0, dtype=torch.float)
    neg = torch.tensor(0, dtype=torch.float)
    loss = torch.tensor(0, dtype=torch.float)

    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[0]):

            if pos_adj_mat[i][j] == 1:
                pos = torch.zeros(1)
                # pos += F.cosine_similarity(x[i].unsqueeze(1), x[j].unsqueeze(0), dim=-1)
                pos += torch.tensor(check(x[i], x[j]), dtype=torch.float)
                pos = torch.mean(pos)
            if neg_adj_mat[i][j] == 1:
                neg = torch.zeros(1)
            #     # neg += F.cosine_similarity(x[i].unsqueeze(1), x[j].unsqueeze(0), dim=-1)
                neg += torch.tensor(check(x[i], x[j]), dtype=torch.float)
                neg = torch.mean(neg)

            loss_1 = -torch.log(torch.exp(pos) / torch.exp(neg + 10e-6))

        loss += loss_1
        loss = torch.mean(loss)
    # 2. 计算距离项,这里使用L2距离,其形状是[BatchSize,BatchSize]
    # x_norm = torch.nn.functional.normalize(x, p=22, dim=1)
    # distance_matrix = torch.cdist(x_norm, x_norm, p=2)
    #
    # # 3.分别计算正负例样本的损失
    # pos_losses = torch.nn.functional.relu(distance_matrix[torch.where(pos_adj_mat)])
    # neg_losses = torch.nn.functional.relu(0.5 - distance_matrix[torch.where(neg_adj_mat)])
    #
    # # 4. 计算总的损失, 分别计算后加起来
    # loss = pos_losses[pos_losses > 0].mean() + neg_losses[neg_losses > 0].mean()
    print("loss:", loss)
    return loss

#对比学习的功效应当是体现在最后对于损失的帮助，也就是全局的，但是由于是批处理提出来的output不对应
optimizer = optim.Adam(model.parameters())

# 将数据转换为TensorDataset和DataLoader
print(X_train)
print(Y_train)
print("&&&")
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train).long())

train_loader = DataLoader(train_dataset, batch_size=seq_len, shuffle=True)

print("%%%%%%%%")
model.train()
loss_list = []
# 训练模型
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs = inputs.squeeze(1)
        outputs = model(inputs)
        # print(outputs)
        # print(outputs.shape)
        # print("565656556")
        optimizer.zero_grad()
        loss = criterion(outputs.data, labels)
        loss.requires_grad_(True)
        #loss = loss+0.0001*contrasive_loss(outputs, labels)

        if epoch % 1000 == 0:
            loss_list.append(loss.item())
        #print(outputs)
        loss.backward()
        optimizer.step()

# 预测验证集的概率
loss_csv = pd.DataFrame(loss_list)
loss_csv.to_csv("loss.csv")

test_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val).long())
test_loader = DataLoader(test_dataset, batch_size=seq_len, shuffle=True)

model.eval()
total_Y_val_prob = []
total_Y_val = []
for inputs, labels in test_loader:
    Y_val_prob = model(inputs.squeeze(1))
    Y_val_prob = Y_val_prob.detach().numpy()
    total_Y_val_prob.extend(Y_val_prob)
    total_Y_val.extend(labels)


    # 计算准确率、查准率、召回率、F1分数和混淆矩阵

Y_val_pred = np.argmax(total_Y_val_prob, axis=1)
total_Y_val_prob = np.array(total_Y_val_prob)
accuracy = accuracy_score(total_Y_val, Y_val_pred)
precision = precision_score(total_Y_val, Y_val_pred)
recall = recall_score(total_Y_val, Y_val_pred)
f1 = f1_score(total_Y_val, Y_val_pred)
confusion = confusion_matrix(total_Y_val, Y_val_pred)

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(total_Y_val, total_Y_val_prob[:, 1])
auc_score = auc(fpr, tpr)

# 打印关键指标和混淆矩阵
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)
print("AUC:", auc_score)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()