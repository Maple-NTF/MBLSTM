import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.nn import functional as F, Module, init
from sklearn.preprocessing import StandardScaler
from transformers import BertModel, BertTokenizer

replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced.csv')
replaced_data_test = pd.read_csv('testdata_0.2_171.csv')#19组
# replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced-20%.csv')
# replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced-40%.csv')
# replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced-60%.csv')
# replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced-80%.csv')
# replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced_5times.csv')
# replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced_refined.csv')
# replaced_data_test = pd.read_csv('testdata_0.2_171_refined.csv')#19组

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
        init.xavier_normal_(self.dense1.weight)
        init.zeros_(self.dense1.bias)#线性层里的参数初始化
        self.dense2 = nn.Linear(units, units)
        init.xavier_normal_(self.dense2.weight)
        init.zeros_(self.dense1.bias)  # 线性层里的参数初始化
        self.dense3 = nn.Linear(units, units)
        init.xavier_normal_(self.dense3.weight)
        init.zeros_(self.dense1.bias)  # 线性层里的参数初始化
        self.dense4 = nn.Linear(units, units)
        init.xavier_normal_(self.dense4.weight)
        init.zeros_(self.dense1.bias)  # 线性层里的参数初始化
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

class LinkPredictor(Module):#线性层预测器，
    def __init__(self, in_feats, n_hidden, out_feats, n_layers,
                 dropout, bn=False):
        super(LinkPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        if bn and n_layers > 1:
            self.bns = torch.nn.ModuleList()
        else:
            self.bns = None
        # 添加linear层
        for i in range(n_layers):
            in_feats_ = n_hidden if i == 0 else n_hidden
            out_feats_ = out_feats if i == n_layers - 1 else n_hidden
            self.lins.append(torch.nn.Linear(in_feats_, out_feats_))
            if bn and i < n_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(out_feats_))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
            init.xavier_normal_(lin.weight)
            lin.bias.data.fill_(0.0)

        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
    def forward(self, x):
        # 拼接

        # print("x", x.size())
        # x = torch.cat([x_i, x_j], dim=-1)
        for i, lin in enumerate(self.lins):
            x = lin(x)
            # print("x_{}",format(i), x.size())
            if i < len(self.lins) - 1:
                if self.bns is not None:
                    x = self.bns[i](x)
                x = torch.relu(x)

        # nn.Softmax(dim=1)
        # 这个是新加上的激活函数
        return x

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_iterations, seq_len):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.seq_len = seq_len

        # 定义前向LSTM单元
        self.forward_lstmcell = nn.LSTMCell(input_size, hidden_size)
        # 定义后向LSTM单元
        self.backward_lstmcell = nn.LSTMCell(input_size, hidden_size)

        # 定义交互模块
        self.interaction_output_forward_h = torch.zeros(1)
        self.interaction_output_backward_h = torch.zeros(1)
        self.interaction_module = InteractionModule(hidden_size, num_iterations)

        # 试图增加正则化内容
        self.parameters_to_regularize = list(self.parameters())  # 添加权重参数到列表

        # 梯度裁剪阈值
        self.clip_value = 0.01

        # 加入批标准化层
        self.batch_norm_forward = nn.BatchNorm1d(hidden_size)
        self.batch_norm_backward = nn.BatchNorm1d(hidden_size)

        # 尝试使用bert作为预训练模型
        # 加载BERT模型和tokenizer
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs, interaction_outputs=None):
        print("inputs:")
        print(inputs.shape)

        # 尝试加入bert，感觉并不是很靠谱，没有正确执行，需要补充内容
        # 使用BERT模型编码输入序列
        # with torch.no_grad():
        #     # 转换输入数据为BERT所需格式
        #     input_ids = inputs["input_ids"]
        #     attention_mask = inputs["attention_mask"]
        #
        #     # 获取BERT的输出
        #     bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #
        #     # 获取BERT的pooler_output用于表示整个句子
        #     bert_output = bert_outputs.pooler_output
        #
        # # 结合BERT的输出和原始输入特征，作为新的输入特征
        # inputs = torch.cat((inputs, bert_output.unsqueeze(0)), dim=1)

        # 初始化前向LSTM隐藏状态和细胞状态
        self.interaction_output_forward_h= torch.zeros(seq_len, self.hidden_size)
        forward_cell = torch.zeros(seq_len, self.hidden_size)

        # 初始化后向LSTM隐藏状态和细胞状态
        self.interaction_output_forward_h = torch.zeros(seq_len, self.hidden_size)
        backward_cell = torch.zeros(seq_len, self.hidden_size)

        # 前向传播
        for t in range(seq_len):
            forward_seq_now = inputs[t]
            backward_seq_now = inputs[seq_len - t - 1]

            # 交互模块
            interaction_output, self.interaction_output_forward_h, self.interaction_output_backward_h = self.interaction_module(
                inputs, self.interaction_output_forward_h, self.interaction_output_backward_h)  # 获取了hf和hb
            interaction_output = (self.interaction_output_forward_h + self.interaction_output_backward_h) / 2
            #interaction_outputs.append(interaction_output)
            # print(self.interaction_output_backward_h.shape)
            # print("464646")

            _, forward_cell[t] = self.forward_lstmcell(forward_seq_now, (self.interaction_output_forward_h[t], forward_cell[t]))
            _, backward_cell[seq_len - t - 1] = self.backward_lstmcell(backward_seq_now, (self.interaction_output_backward_h[seq_len - t - 1], backward_cell[seq_len - t - 1]))

            # 试图加入批标准化的内容
            self.interaction_output_forward_h = self.batch_norm_forward(self.interaction_output_forward_h)
            self.interaction_output_backward_h = self.batch_norm_backward(self.interaction_output_backward_h)

        print("interaction_output:")
        print(interaction_output.shape)

        # 试图加入梯度裁剪前向LSTM单元
        # nn.utils.clip_grad_norm_(self.forward_lstmcell.parameters(), self.clip_value)
        # # 梯度裁剪后向LSTM单元
        # nn.utils.clip_grad_norm_(self.backward_lstmcell.parameters(), self.clip_value)

        # 将交互模块的输出转换为张量
        return interaction_output

        # 尝试使用残差连接缓解梯度消失问题
        # interaction_output = []  # 存储多个 Tensor 的列表
        # for output in interaction_output:
        #     interaction_output.append(output)
        # if interaction_output:
        #     interaction_output = torch.stack(interaction_output)
        #     # 添加残差连接
        #     residual_output = inputs + interaction_output
        # else:
        #     residual_output = inputs
        #
        # return residual_output

# 定义模型参数
input_size = X_train.shape[2]
print("input_size:" ,input_size)#15
hidden_size = 15#15#64
seq_len = 9
num_iterations = 1 #2交互模块迭代
num_classes = 15#2
batch_size = 30#30
num_epochs = 300#300

# 定义模型
model = nn.Sequential(
    BiLSTM(input_size, hidden_size, num_iterations, seq_len),
    # nn.Dropout(0.2),
    # nn.Linear(hidden_size * 1, hidden_size),#2
    # nn.Dropout(0.2),
    # nn.ReLU(),
    # nn.Sigmoid(),
    # nn.Dropout(0.2),
    # nn.Linear(hidden_size, num_classes),
    # nn.Dropout(0.2),
    # nn.Softmax(dim=-1),
    # nn.Sigmoid(),
    nn.ReLU(),
    # nn.Sigmoid()
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
# criterion = nn.BCELoss()#二元交叉熵损失
# criterion = nn.MSELoss()#均方差损失
# criterion = nn.KLDivLoss()#KL散度损失

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
                pos += check(x[i], x[j]).clone().detach().requires_grad_(True)
                pos = torch.mean(pos)
            if neg_adj_mat[i][j] == 1:
                neg = torch.zeros(1)
            #     # neg += F.cosine_similarity(x[i].unsqueeze(1), x[j].unsqueeze(0), dim=-1)
                neg += check(x[i], x[j]).clone().detach().requires_grad_(True)
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

predictor = LinkPredictor(num_classes, num_classes, 2, 2, None)
# predictor = LinkPredictor(15, 15, 2, 2, None)

parameters = list(model.parameters()) + list(predictor.parameters())

#对比学习的功效应当是体现在最后对于损失的帮助，也就是全局的，但是由于是批处理提出来的output不对应
# optimizer = optim.Adam(parameters, lr = 0.00005)
# optimizer = optim.Adam(parameters,lr = 0.000005, weight_decay=0.01)#加入正则化
optimizer = optim.SGD(parameters,lr = 0.05, momentum = 0.9)#配合one circle
# optimizer = optim.RMSprop(parameters,lr = 0.000005)

# 尝试增加余弦退火的内容，定义余弦退火学习率调度器,T_max表示一个完整的训练周期的步数,eta_min表示学习率的下界
# scheduler = CosineAnnealingLR(optimizer, T_max=9, eta_min=0.001)

# 尝试加入 one circle方法，是综合了余弦退火和动量的方法
scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps = num_epochs * 121)

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
    # 在每个epoch开始前调用一次step()来更新学习率，余弦退火
    # scheduler.step()
    for inputs, labels in train_loader:
        inputs = inputs.squeeze(1)
        # # 数据标准化
        # # 创建一个标准化器对象
        # scaler = StandardScaler()
        # # 对数据进行标准化
        # inputs = torch.tensor(scaler.fit_transform(inputs), dtype=torch.float32)
        outputs = model(inputs)
        print(outputs)
        # print(outputs)
        # print(outputs.shape)
        # print("565656556")
        optimizer.zero_grad()
        loss = criterion(outputs.data, labels)
        # loss = loss + 0.001 * contrasive_loss(outputs, labels)
        loss.requires_grad_(True)


        # 测试部分
        test_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val).long())
        test_loader = DataLoader(test_dataset, batch_size=seq_len, shuffle=True)

        # model.eval()
        # total_Y_val_prob = []
        # total_Y_val = []
        # # 为测试数据增加一个和训练数据相同的循环
        # # 调整数据格式以及循环格式
        # # for inputs, labels in train_loader:
        # for inputs, labels in test_loader:
        #     predictor.eval()
        #     outputs_final = model(inputs.squeeze(1))
        #     Y_val_prob = predictor(outputs_final)
        #
        #     Y_val_prob = Y_val_prob.detach().numpy()
        #
        #     print("@@@@@@")
        #     print("Y_val_prob:")
        #     print(Y_val_prob)
        #     total_Y_val_prob.extend(Y_val_prob)
        #     total_Y_val.extend(labels)
        #
        # # 计算准确率、查准率、召回率、F1分数和混淆矩阵
        # Y_val_pred = np.argmax(total_Y_val_prob, axis=1)
        # total_Y_val_prob = np.array(total_Y_val_prob)
        # accuracy = accuracy_score(total_Y_val, Y_val_pred)
        # precision = precision_score(total_Y_val, Y_val_pred)
        # recall = recall_score(total_Y_val, Y_val_pred)
        # f1 = f1_score(total_Y_val, Y_val_pred)
        # confusion = confusion_matrix(total_Y_val, Y_val_pred)
        #
        # # 计算ROC曲线和AUC值
        # fpr, tpr, thresholds = roc_curve(total_Y_val, total_Y_val_prob[:, 1])
        # auc_score = auc(fpr, tpr)
        #
        # # 打印关键指标和混淆矩阵
        # print("Accuracy:", accuracy)
        # print("Precision:", precision)
        # print("Recall:", recall)
        # print("F1 Score:", f1)
        # print("Confusion Matrix:")
        # print(confusion)
        # print("AUC:", auc_score)
        # model.train()

        if epoch / 1000 == 0:
            loss_list.append(loss.item())
        #print(outputs)
        loss.backward()
        optimizer.step()
        scheduler.step()#配合one circle

# 预测验证集的概率
# 打印损失
loss_csv = pd.DataFrame(loss_list)
loss_csv.to_csv("loss.csv")

#测试部分
test_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val).long())
test_loader = DataLoader(test_dataset, batch_size=seq_len, shuffle=True)

model.eval()
total_Y_val_prob = []
total_Y_val = []
#为测试数据增加一个和训练数据相同的循环
#调整数据格式以及循环格式
# for inputs, labels in train_loader:
for inputs, labels in test_loader:
    predictor.eval()
    outputs_final = model(inputs.squeeze(1))
    Y_val_prob = predictor(outputs_final)

    Y_val_prob = Y_val_prob.detach().numpy()

    print("@@@@@@")
    print("Y_val_prob:")
    print(Y_val_prob)
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