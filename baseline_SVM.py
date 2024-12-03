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

replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_new_replaced.csv')
replaced_data_test = pd.read_csv('testdata_0.2_171.csv')#19组

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


from sklearn.svm import SVC

# 将输入数据转换为2D张量
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[2]))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[2]))

# 初始化SVM模型
svm_model = SVC()

# 训练SVM模型
svm_model.fit(X_train, Y_train)

# 在验证集上进行预测
predicted_labels = svm_model.predict(X_val)

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