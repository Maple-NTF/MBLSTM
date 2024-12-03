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

# 从CSV文件中读取训练数据
# data = pd.read_csv('data2023.08.30.csv')#使用了倒置之后的数据
# resampled_data = pd.read_csv('enhanced_df_replaced.csv')


# 将class列的数据替换为1和0
# data['class'] = data['class'].replace({'IntervalPeriod': 1, 'AcutePhase': 0})
# #data = data.astype('float64')
# print(data)
# resampled_data['class'] = resampled_data['class'].replace({'IntervalPeriod': 1, 'AcutePhase': 0})
# resampled_data = resampled_data.astype('float64')
# print(resampled_data.head())
#试图使用对比学习方法来增加负例目前还没有调试完成
# 提取特征和标签
# X = resampled_data.iloc[:, :-1].values
# y = resampled_data.iloc[:, -1].values
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# 使用SMOTE进行数据增强
# smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X, y)

# 将增强后的数据保存到新的CSV文件
# resampled_data = pd.DataFrame(X_resampled, columns=data.columns[:-1])
# resampled_data['label'] = y_resampled
# resampled_data.to_csv('resampled_data_smote.csv', index=False)
# print(resampled_data.shape)
# print(resampled_data.head())
#
# resampled_data = resampled_data.astype('float64')

# # X_train = data.iloc[:, :-1].values
# # Y_train = data.iloc[:, -1].values
# X_train = resampled_data.iloc[:, :-1].values
# print(X_train[:5])
# Y_train = resampled_data.iloc[:, -1].values
# print(Y_train[:5])
# X_train = X_train.astype('float64')
# #为了解决Y_train的问题
# # 创建LabelEncoder对象
# label_encoder = LabelEncoder()
# # 对Y_train进行编码
# Y_train_encoded = label_encoder.fit_transform(Y_train)
# # 将编码后的Y_train转换为float类型
# Y_train = Y_train_encoded.astype('float64')
# #Y_train = Y_train.astype('float64')#之前是str
# # print("####")
# # print(Y_train)
# # print("####")

#划分验证集
# from sklearn.model_selection import train_test_split
# # 划分训练集和验证集
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=35)

# 将输入数据转换为3D张量
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
# # print("####")
# # print(X_train)
# # print("####")
# # print("####")
# # print(X_val)
# # print("####")


#全复制的时候应该是有77行代码，此时的代码是没有针对0.2和0.8的训练测试集划分的，且是没有将写死了的训练集和测试集的位置进行调整的
#现在的工作是，首先把划分训练集和测试集的语句分开
# 从CSV文件中读取训练数据
replaced_data_train = pd.read_csv('0.8_enhanced_df_withclass_replaced.csv')
replaced_data_test = pd.read_csv('0.2_testdata.csv')

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
smote = SMOTE()
X_smoted, Y_smoted = smote.fit_resample(X, Y)

# 将增强后的数据保存到新的CSV文件
augmented_data = pd.DataFrame(X_smoted, columns=replaced_data_train.columns[:-1])
augmented_data['label'] = Y_smoted
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

#全复制的时候应该是有77行代码，此时的代码是没有针对0.2和0.8的训练测试集划分的，且是没有将写死了的训练集和测试集的位置进行调整的
#现在的工作是，首先把划分训练集和测试集的语句分开
# 从CSV文件中读取训练数据

#已经没有划分训练集和验证集的步骤了
#但是验证集的数据仍然需要转换成正确的形式进行训练
#现在的目标是创造出和Xtrain与Ytrain相同的Xval与Yval，也就是把测试集转换成一样的
#现在是按照训练集的模样把测试集整出来了


