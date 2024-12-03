import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# 从CSV文件中读取训练数据
data = pd.read_csv('data2023.08.30.csv')
print(data.head())
X_train = data.iloc[:, :-1].values
print(X_train[:5])
Y_train = data.iloc[:, -1].values
print(Y_train[:5])
X_train = X_train.astype('float64')

# 创建LabelEncoder对象
label_encoder = LabelEncoder()
# 对Y_train进行编码
Y_train_encoded = label_encoder.fit_transform(Y_train)
# 将编码后的Y_train转换为float类型
Y_train = Y_train_encoded.astype('float64')

#划分验证集
from sklearn.model_selection import train_test_split
# 划分训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=35)

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
        print(x0)
        hf = forward_hidden_state
        print(hf)
        hb = backward_hidden_state
        print(hb)

        for _ in range(self.num_iterations):

            x1 = self.sigmoid(self.dense1(x0 + hf))
            hb2 = self.sigmoid(self.dense2(hb + x1))
            hf2 = self.sigmoid(self.dense3(x1 + hf))
            x2 = self.sigmoid(self.dense4(hb2 + x1))

            x0 = x2
            hf = hf2
            hb = hb2

        return x0, hf, hb

# 定义双向LSTM层
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_iterations):
        super(BidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_iterations = num_iterations
        self.forward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        print("input_size（after LSTM）:" ,input_size)
        print("hidden_size（after LSTM）:", hidden_size)
        self.interaction_module = InteractionModule(hidden_size, num_iterations)

    def forward(self, inputs):
        forward_output, _ = self.forward_lstm(inputs)
        backward_output, _ = self.backward_lstm(inputs.flip(dims=[1]))
        backward_output = backward_output.flip(dims=[1])

        forward_h, forward_c = forward_output[:, :, :self.hidden_size], forward_output[:, :, self.hidden_size:]
        backward_h, backward_c = backward_output[:, :, :self.hidden_size], backward_output[:, :, self.hidden_size:]


        interaction_output, interaction_output_forward_h, interaction_output_backward_h = self.interaction_module(inputs, forward_h, backward_h)#获取了hf和hb

        output_h = (interaction_output_forward_h + interaction_output_backward_h) / 2

        combined_output = torch.cat([interaction_output, output_h], dim=1)
        combined_output = torch.mean(combined_output, dim=1)

        print("combined_output:", combined_output)
        return combined_output


# 定义模型参数
input_size = X_train.shape[2]
print("input_size:" ,input_size)
hidden_size = 15
num_iterations = 2
num_classes = 2
batch_size = 2
num_epochs = 13

# 定义模型
model = nn.Sequential(
    BidirectionalLSTM(input_size, hidden_size, num_iterations),
    nn.Linear(hidden_size * 1, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, num_classes),
    nn.Softmax(dim=-1)
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 将数据转换为TensorDataset和DataLoader
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train).long())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 预测验证集的概率
Y_val_prob = model(torch.Tensor(X_val))
Y_val_prob = Y_val_prob.detach().numpy()

# 计算准确率、召回率、F1分数和混淆矩阵
Y_val_pred = np.argmax(Y_val_prob, axis=1)
accuracy = accuracy_score(Y_val, Y_val_pred)
recall = recall_score(Y_val, Y_val_pred)
f1 = f1_score(Y_val, Y_val_pred)
confusion = confusion_matrix(Y_val, Y_val_pred)

# 计算ROC曲线和AUC值
fpr, tpr, thresholds = roc_curve(Y_val, Y_val_prob[:, 1])
auc_score = auc(fpr, tpr)

# 打印关键指标和混淆矩阵
print("Accuracy:", accuracy)
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

#旨在全部做完之后做一个清爽的版本，目前还没做完
