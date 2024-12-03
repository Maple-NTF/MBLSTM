import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.neighbors import KernelDensity
from scipy import stats

data = pd.read_csv("9-334data.csv")

# 设置保存路径
save_path = "E:\9-334data数据图片"

#matplotlib.use('Agg')

# 获取自定义ID列
custom_id = data['custom_id']

# 获取变量列
variables = data.drop('custom_id', axis=1)

# 生成15张图
# 遍历每一列进行核密度估计和绘图
# for variable in variables.columns:
#     # 创建子图
#     fig, ax = plt.subplots(figsize=(6, 4))
#
#     # 获取当前变量的数据
#     x = variables[variable].values.reshape(-1, 1)
#
#     # 创建核密度估计模型
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x)
#
#     # 生成一组用于绘制估计曲线的数据
#     x_plot = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
#     log_dens = kde.score_samples(x_plot)
#
#     # 绘制核密度估计曲线
#     ax.plot(x_plot, np.exp(log_dens))
#     ax.set_title(variable)
#     ax.set_xlabel('Value')
#     ax.set_ylabel('Density')
#
#     file_name = f'{variable}.png'
#     file_path = os.path.join(save_path, file_name)
#     fig.savefig(file_path)
#     plt.close(fig)
#     # plt.show()

#生成1张图包含15个子图
# 创建子图
# fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(12, 12))
#
# # 遍历每一列进行核密度估计和绘图
# for i, variable in enumerate(variables.columns):
#     row = i // 4
#     col = i % 4
#
#     # 获取当前变量的数据
#     x = variables[variable].values.reshape(-1, 1)
#
#     # 创建核密度估计模型
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x)
#
#     # 生成一组用于绘制估计曲线的数据
#     x_plot = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
#     log_dens = kde.score_samples(x_plot)
#
#     # 绘制核密度估计曲线
#     axs[row, col].plot(x_plot, np.exp(log_dens))
#     axs[row, col].set_title(variable)
#     axs[row, col].set_xlabel('Value')
#     axs[row, col].set_ylabel('Density')
#
# # 调整子图之间的间距
# plt.tight_layout()
#
# # 显示图形
# plt.show()


# 一图4法 但是较为混乱
# # 遍历每一列进行核密度估计和绘图
# for variable in variables.columns:
#     # 创建子图
#     fig, ax = plt.subplots(figsize=(6, 4))
#
#     # 获取当前变量的数据
#     x = variables[variable].values.reshape(-1, 1)
#
#     # 创建核密度估计模型
#     for kernel in ['gaussian', 'tophat', 'epanechnikov', 'exponential']:
#         for bandwidth in [0.1, 0.5, 1.0, 2.0]:
#             kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(x)
#
#             # 生成一组用于绘制估计曲线的数据
#             x_plot = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
#             log_dens = kde.score_samples(x_plot)
#
#             # 绘制核密度估计曲线
#             ax.plot(x_plot, np.exp(log_dens), label=f'{kernel}, bw={bandwidth}')
#
#     ax.set_title(variable)
#     ax.set_xlabel('Value')
#     ax.set_ylabel('Density')
#     ax.legend()
#
#     # 显示图形
#     plt.show()



# 遍历每一列进行核密度估计和绘图
for variable in variables.columns:
    # 创建子图
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    #fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))

    # 获取当前变量的数据
    x = variables[variable].values.reshape(-1, 1)

    # 不同核函数和带宽参数的选项
    kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential']
    #bandwidths = [0.1, 0.5, 1.0, 2.0]
    #bandwidths = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    bandwidth_scott = stats.iqr(x) / (1.34 * len(x) ** 0.2)
    bandwidth_silverman = 0.9 * min(np.std(x), stats.iqr(x) / 1.34) * len(x) ** (-0.2)

    # 遍历不同核函数和带宽参数的选项
    for i, kernel in enumerate(kernels):
        row = i // 2
        col = i % 2

        # 创建核密度估计模型
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth_scott).fit(x)
        #kde = KernelDensity(kernel=kernel, bandwidth=bandwidth_silverman).fit(x)

        # 生成一组用于绘制估计曲线的数据
        x_plot = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
        log_dens = kde.score_samples(x_plot)

        # 绘制核密度估计曲线
        axs[row, col].plot(x_plot, np.exp(log_dens))
        axs[row, col].set_title(f'{kernel}, bw={bandwidth_scott}')
        #axs[row, col].set_title(f'{kernel}, bw={bandwidth_silverman}')
        axs[row, col].set_xlabel('Value')
        axs[row, col].set_ylabel('Density')

    fig.suptitle(variable)
    plt.tight_layout()

    file_name = f'{variable}.png'
    file_path = os.path.join(save_path, file_name)
    fig.savefig(file_path)
    plt.close(fig)

    # 显示图形
    plt.show()


