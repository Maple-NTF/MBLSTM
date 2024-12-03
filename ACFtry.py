import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 输入数据
# data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]] * 37)
data = pd.read_csv("9-334data.csv")
data.head()
print(data.head())
print(data.shape)

# 获取标签和数据
labels = data.columns[1:]
print("labels:",labels)
custom_ids = data.iloc[:, 0]

# matplotlib.use('Agg')

#15张每列一条折线数据

# # 设置保存路径
# save_path = "E:\9-334data数据图片"
#
# # 遍历每一列数据，生成图表
# for label in labels:
#     plt.figure()
#     plt.plot(custom_ids, data[label])
#     plt.xlabel('custom_id')
#     plt.ylabel(label)
#     plt.title(f'{label} vs. custom_id')
#     plt.grid(True)
#     plt.show()
#     plt.savefig(os.path.join(save_path, f'{label}_vs_custom_id.png'))

# 计算自相关系数
correlations = {}
for label in labels:
    correlation = data[label].autocorr()
    correlations[label] = correlation

# 打印自相关系数
print("ACF:")
for label, correlation in correlations.items():
    print(f'{label}: {correlation}')

# 计算自协方差
covariances = {}
for label in labels:
    covariance = data[label].cov(data[label])
    covariances[label] = covariance

# 打印自协方差
print("ACVF:")
for label, covariance in covariances.items():
    print(f'{label}: {covariance}')

# 找出超出自协方差3倍的数据
outliers = {}
for label in labels:
    column = data[label]
    threshold = 3 * covariances[label]
    outlier_indices = column[abs(column - column.mean()) > threshold].index
    outliers[label] = data.loc[outlier_indices]

# 打印超出自协方差3倍的数据
print("Exceed 3 times:")
for label, outlier_data in outliers.items():
    print(f'{label}:')
    print(outlier_data)
    print()

#15张每列三十七条折线数据

# 设置保存路径
save_path = "E:\9-334data数据图片"

# 将数据分组
grouped_data = [data.iloc[i:i+9] for i in range(0, len(data), 9)]

# 绘制折线图
for i, label in enumerate(labels):
    fig, ax = plt.subplots()
    for group in grouped_data:
        ax.plot(range(1, 10), group[label])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Group {i+1}')
    ax.legend()
    # plt.show()
    file_name = f'{label}.png'
    file_path = os.path.join(save_path, file_name)
    fig.savefig(file_path)
    plt.close(fig)

























# 调整数据形状
# data = data.reshape(-1, 1)
# print(data)
#
# # 创建核密度估计模型
# kde = KernelDensity()
#
# # 拟合数据
# kde.fit(data)
#
# # 生成新数据点
# new_data = np.linspace(1, 9, 100).reshape(-1, 1)
#
# # 计算核密度估计值
# density_estimation = np.exp(kde.score_samples(new_data))
#
# # 打印结果
# print(density_estimation)
# print(density_estimation.size)