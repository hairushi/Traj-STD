import os

import torch
import torch.nn as nn
from torch import optim, cuda
from torch.autograd import Variable
from tqdm import tqdm

from data_load import get_real_data, get_syn_data
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyCombinedDataset(Dataset):
    def __init__(self, real_start_loc, real_start_time, syn_start_loc):
        self.real_start_loc = real_start_loc
        self.real_start_time = real_start_time
        self.syn_start_loc = syn_start_loc

    def __len__(self):
        return len(self.real_start_loc)

    def __getitem__(self, idx):
        return {
            'real_start_loc': self.real_start_loc[idx],
            'real_start_time': self.real_start_time[idx],
            'syn_start_loc': self.syn_start_loc[idx]
        }


# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=3)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, c, x):
        out, hidden = self.gru(torch.cat((c, x), 2))
        out = self.linear(out)
        return out


# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=3)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, x):
        out, hidden = self.gru(torch.cat((c, x), 2))
        out = self.linear(out)
        out = self.sigmoid(out)
        return out


# 定义训练函数
def train(input_size, hidden_size, seq_len, batch_size, lr, num_epochs, real_data_file, syn_data_file):
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # 加载数据
    real_start_loc, real_start_time = get_real_data(real_data_file)
    syn_start_loc = get_syn_data(syn_data_file)
    real_start_loc_reshaped = real_start_loc.reshape(-1, real_start_loc.shape[-1])
    real_start_time_reshaped = real_start_time.reshape(-1, real_start_time.shape[-1])
    syn_start_loc_reshaped = syn_start_loc.reshape(-1, syn_start_loc.shape[-1])

    # 最大最小归一化
    mm_real_start_loc = MinMaxScaler()
    mm_real_start_time = MinMaxScaler()
    mm_syn_start_loc = MinMaxScaler()
    real_start_loc_m = mm_real_start_loc.fit_transform(real_start_loc_reshaped).reshape(real_start_loc.shape)
    real_start_time_m = mm_real_start_time.fit_transform(real_start_time_reshaped).reshape(real_start_time.shape)
    syn_start_loc_m = mm_syn_start_loc.fit_transform(syn_start_loc_reshaped).reshape(syn_start_loc.shape)

    # 创建组合数据集实例
    combined_dataset = MyCombinedDataset(real_start_loc_m, real_start_time_m, syn_start_loc_m)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 初始化生成器和判别器
    generator = Generator(input_size, hidden_size)
    discriminator = Discriminator(input_size, hidden_size)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        criterion.cuda()

    for epoch in range(num_epochs):
        for batch in tqdm(dataloader):
            real_start_loc_batch = batch['real_start_loc'].to(torch.float32).to(device)
            real_start_time_batch = batch['real_start_time'].to(torch.float32).to(device)
            syn_start_loc_batch = batch['syn_start_loc'].to(torch.float32).to(device)
            # 生成噪音
            noise = torch.randn(batch_size, seq_len, 1).to(device)

            # 生成验证矩阵
            valid = Variable(FloatTensor(batch_size, seq_len, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, seq_len, 1).fill_(0.0), requires_grad=False)

            # 训练生成器
            generator.zero_grad()
            fake_time = generator(syn_start_loc_batch, noise)
            validity = discriminator(syn_start_loc_batch, fake_time)
            g_loss = criterion(validity, valid)
            g_loss.backward()
            optimizer_g.step()

            # 训练判别器
            discriminator.zero_grad()
            validity_real = discriminator(real_start_loc_batch, real_start_time_batch)
            d_real_loss = criterion(validity_real, valid)

            validity_fake = discriminator(syn_start_loc_batch, fake_time.detach())
            d_fake_loss = criterion(validity_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            if (epoch + 1) % 10 == 0:
                gen_time = generator(syn_start_loc_batch, noise)
                numpy_array = gen_time.cpu().detach().numpy()
                numpy_array_reshaped = numpy_array.reshape(-1, real_start_time.shape[-1])
                gen_time = mm_real_start_time.inverse_transform(numpy_array_reshaped)
                gen_time = gen_time.reshape(numpy_array.shape)
                # filename = os.path.join('./gen_time', f"epoch_{epoch + 1}.txt")
                # with open(filename, 'w') as f:
                #     f.write(str(gen_time) + "\n")

        if (epoch + 1) % 1 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}')

    torch.save(generator, './model/generator/generator_2w_120s_b4_LTS.pth')
    print('成功保存文件模型文件！')


if __name__ == '__main__':
    input_size = 3  # 输入维度，假设是轨迹的坐标
    hidden_size = 128  # 隐藏层维度
    output_size = 1  # 输出维度
    lr = 0.001  # 学习率
    num_epochs = 1000
    batch_size = 64
    seq_len = 5
    real_file = './data/porto_2w_iter120_b4.txt'
    syn_file = './data/two_dim_porto_2w_iter120_b4_1.txt'
# 训练模型
    train(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len,
          batch_size=batch_size, lr=lr, real_data_file=real_file, syn_data_file=syn_file,
          num_epochs=num_epochs)
