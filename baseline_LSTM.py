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
from torch.nn import functional as F, Module, init
from sklearn.preprocessing import StandardScaler

# replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced.csv')
# replaced_data_test = pd.read_csv('testdata_0.2_171.csv')#19组
replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced_refined.csv')
replaced_data_test = pd.read_csv('testdata_0.2_171_refined.csv')#19组

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
# print("####")
# print(X_train)
# print("####")
# print("####")
# print(X_val)
# print("####")

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 设置超参数
input_size = X_train.shape[2]
hidden_size = 15
num_layers = 32
output_size = 1
num_epochs = 500
batch_size = 9
learning_rate = 0.0001

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 将数据转换为TensorDataset和DataLoader
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印训练过程中的损失
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在验证集上进行预测
model.eval()
with torch.no_grad():
    inputs = torch.from_numpy(X_val).float()
    labels = torch.from_numpy(Y_val).float()
    outputs = model(inputs)
    predicted_labels = torch.round(torch.sigmoid(outputs)).cpu().numpy()

# 计算关键指标和混淆矩阵
accuracy = accuracy_score(Y_val, predicted_labels)
precision = precision_score(Y_val, predicted_labels)
recall = recall_score(Y_val, predicted_labels)
f1 = f1_score(Y_val, predicted_labels)
confusion = confusion_matrix(Y_val, predicted_labels)


print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(confusion)

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(Y_val, predicted_labels)
roc_auc = auc(fpr, tpr)
auc_score = auc(fpr, tpr)
print("AUC:", auc_score)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

