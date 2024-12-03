import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# 输入数据
data = pd.read_csv("896x0.8_for_KDE_new.csv")
df = pd.read_csv('896x0.8_for_KDE_new.csv')
# 对除了第一列以外的数据进行自然对数操作（防止负数）
df.iloc[:, 1:] = np.log(df.iloc[:, 1:])
print(df)

# 定义Expanechnikov核函数
def epanechnikov_kernel(u):
    return 0.75 * (1 - u ** 2) * (abs(u) <= 1)

# 定义一个函数来应用Epanechnikov核的KDE
def kde_epanechnikov(data, bandwidth):
    kde = gaussian_kde(data, bw_method=bandwidth)
    # 重写kde的evaluate方法，以使用Epanechnikov核
    def epanechnikov_evaluate(u):
        # 将u转换为数据空间
        u = (u - kde.dataset) / kde.factor
        result = np.sum(epanechnikov_kernel(u), axis=1) / kde._norm_factor
        return result
    kde.evaluate = epanechnikov_evaluate
    return kde

# 计算Silverman带宽因子
def silverman_bandwidth(data):
    return 1.06 * np.std(data) * len(data) ** (-1 / 5)

# 根据custom_id分组，然后对每组数据应用KDE
groups = df.groupby('custom_id')
# 遍历每个分组并打印
# for name, group in groups:
#     print(f"Group: {name}")
#     print(group)
#     print("\n")
# print(groups.size())
# print('-------------')

# 准备一个空的DataFrame来存放增强后的数据
enhanced_df = pd.DataFrame(columns=df.columns)
# print(enhanced_df)
# print('-------------')

for name, group in groups:
    # 如果组内行数不足9行，则进行增强
    if len(group) < 9:
        # 创建一个临时的DataFrame来存储增强后的数据
        temp_df = pd.DataFrame(index=range(9), columns=group.columns)
        temp_df['custom_id'] = name  # 设置 custom_id 列的值
        print(temp_df)
        print('-------------')
        new_samples = 9 - len(group)
        # 对每列进行核密度估计，除了custom_id列
        for col in group.columns[1:]:
            print(col)
            coldata = group[col].dropna().values  # 确保去除NaN值
            # print(group[col])
            # print('+++++++++++')
            bandwidth = silverman_bandwidth(coldata)
            # 使用重写的ExponentialKDE来进行估计
            # print(coldata)
            # print('+++++++++++')
            kde = kde_epanechnikov(coldata, bandwidth)
            # 生成增强的数据
            new_data = kde.resample(new_samples).flatten()
            # print(new_data)
            # print('+++++++++++')
            # 将原数据和新数据合并，忽略原始数据中的NaN值
            combined_data = np.concatenate([coldata, new_data])
            print(combined_data)
            print('***********')
            # 创建一个新列，如果原始数据中有NaN，先填充NaN，然后再赋值
            temp_df[col] = np.nan
            temp_df.loc[:len(combined_data) - 1, col] = combined_data
            print(temp_df)
            print('///////////')

        # 将处理后的数据加入到增强后的 DataFrame 中
        enhanced_df = enhanced_df.append(temp_df, ignore_index=True)
    else:
        # 如果行数足够，则直接添加到增强后的 DataFrame
        enhanced_df = enhanced_df.append(group, ignore_index=True)

# 打印增强数据（取了ln的）
print(enhanced_df)

#取指数转回来并打印增强数据
enhanced_df.iloc[:, 1:] = np.exp(enhanced_df.iloc[:, 1:])
print(enhanced_df)

# 输出增强后的数据到CSV文件
enhanced_df.to_csv('0.8_enhanced_df_withclass_new.csv', index=False)



