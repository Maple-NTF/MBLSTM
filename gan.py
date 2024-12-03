import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

# 读取CSV数据
data = pd.read_csv('data2023.08.30.csv')

# 提取特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# 创建LabelEncoder对象
label_encoder = LabelEncoder()
# 转换为PyTorch的Tensor格式
X_tensor = torch.tensor(X, dtype=torch.float64)
# y_tensor = torch.tensor(y, dtype=torch.float64)
# 对y进行编码
Y_encoded = label_encoder.fit_transform(y)
# 将编码后的y_encoded转换为float类型
Y_tensor = Y_encoded.astype('float64')

# print("X_tensor_g:",X_tensor)
# print("++++++++++")
# print("Y_tensor_g:",Y_tensor)

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        return x, y

# 创建数据集对象
dataset = MyDataset(X_tensor, Y_tensor)

# 创建数据集和数据加载器
# dataset = TensorDataset(X_tensor, Y_tensor)问题始终无法解决
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, input_dim),#1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# 定义GRU模型
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gru = nn.GRU(self.input_dim, self.hidden_dim,1)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        _, hidden = self.gru(x)
        # print("hidden:",hidden)
        # print("+++++")
        output = self.fc(hidden.squeeze(0))
        return output


# 定义训练函数
def train_timegan(generator, discriminator, gru, dataloader, num_epochs, lr):
    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss(reduction='sum')
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    # Train Stage 1: Training GAN
    for epoch in range(num_epochs):
        for i, (real_data, _) in enumerate(dataloader):
            batch_size = real_data.size(0)

            # 训练判别器
            optimizer_d.zero_grad()
            real_labels = torch.ones(batch_size, input_dim)#1
            fake_labels = torch.zeros(batch_size, input_dim)#1

            real_output = discriminator(real_data)
            real_loss = criterion_bce(real_output, real_labels)

            z = torch.randn(batch_size, input_dim)
            fake_data = generator(z)
            fake_output = discriminator(fake_data.detach())
            fake_loss = criterion_bce(fake_output, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = criterion_bce(fake_output, real_labels)
            g_loss.backward()
            optimizer_g.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")

    # Train Stage 2: Training GRU
    optimizer_gru = optim.Adam(gru.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i, (real_data, _) in enumerate(dataloader):
            real_data = real_data
            #维度从(batch_size, sequence_length, input_dim)扩展为(batch_size, sequence_length, 1, input_dim)。这样做的目的是为了与GRU模型的输入要求相匹配。
            print("------------------")
            print("real_data:",real_data)
            print("------------------")
            print("real_data.shape:",real_data.shape)

            # 训练GRU
            optimizer_gru.zero_grad()

            z = torch.randn(batch_size, input_dim)
            fake_data = generator(z)

            real_prediction = gru(real_data)
            fake_prediction = gru(fake_data)

            mse_loss = criterion_mse(fake_prediction, real_prediction)
            mse_loss.backward()
            optimizer_gru.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], GRU Loss: {mse_loss.item():.4f}")


# 定义TimeGAN模型的参数
input_dim = X.shape[1]
print("input_dim:",input_dim)
print("-----------------")
output_dim = X.shape[1]
print("output_dim:",output_dim)
print("-----------------")
hidden_dim = X.shape[1]
print("hidden_dim:",hidden_dim)
print("-----------------")
num_epochs = 10
lr = 0.0002

# 创建生成器、判别器和GRU模型
generator = Generator(input_dim, output_dim)
print("generator:",generator)
print("-----------------")
discriminator = Discriminator(input_dim)
print("discriminator:",discriminator)
print("-----------------")
gru = GRU(output_dim, hidden_dim, output_dim)
print("gru:",gru)
print("-----------------")

# 训练TimeGAN模型
train_timegan(generator, discriminator, gru, dataloader, num_epochs, lr)

# 生成增强后的数据
noise_c = torch.randn(X.shape[0], input_dim)
generated_data = generator(noise_c).detach().numpy()

# 将增强后的数据与原始数据合并
augmented_data = np.concatenate([X, generated_data])
print("augmented_data:",augmented_data)
print("++++++++++++++")
print("augemented_data.size:",augmented_data.size)
print("++++++++++++++")
print("augemented_data.shape:",augmented_data.shape)
augmented_labels = np.concatenate([y, np.ones((X.shape[0]))])

# 将增强后的数据保存到新的CSV文件
resampled_data = pd.DataFrame(augmented_data, columns=data.columns[:-1])
resampled_data['label'] = augmented_labels
resampled_data.to_csv('resampled_data_TimeGAN.csv', index=False)



# input_dim: 数据的特征维度。在时间序列数据中，它代表每个时间步的特征数量。
#
# hidden_dim: LSTM 网络中隐藏层的维度。它决定了 LSTM 网络的复杂程度和能力。较大的 hidden_dim 可能会导致更复杂的模型，但可能需要更多的计算资源。
#
# output_dim: 生成器模型的输出维度。通常情况下，它与时间序列数据的特征维度相同。
#
# num_epochs: 训练的总轮数，即迭代次数。您可以根据数据集的大小和复杂性来设置合适的值。
#
# lr: 学习率，用于优化器的学习速率。学习率的选择对训练的效果有很大影响，通常需要进行调试和优化。
#
# batch_size: 每个训练批次的样本数量。较大的批次大小可以加快训练速度，但可能会导致内存不足。较小的批次大小可以提高模型的稳定性，但训练速度较慢。
#
# criterion: 损失函数，用于计算生成器和判别器的损失。在 TimeGAN 中，可以使用适合时间序列数据的损失函数，如均方误差（MSE）或 GAN 的损失函数。
#
# optimizer_g 和 optimizer_d: 生成器和判别器的优化器，分别用于更新生成器和判别器的参数。在代码中使用了 Adam 优化器 optim.Adam() 。
#
# real_labels 和 fake_labels: 真实样本和生成样本的标签。在 GAN 中，通常使用全1和全0的标签来区分真实样本和生成样本。
#
# real_output 和 fake_output: 判别器对真实样本和生成样本的输出。用于计算判别器的损失。