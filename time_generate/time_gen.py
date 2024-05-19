import os
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_load import get_real_data, get_all_syn_data, get_syn_data
from utils import formalize_data
from time_change import change_time


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用 map_location 将模型加载到 CPU 上
load_model = torch.load("./model/generator/generator_2w_120s_b4_LTS.pth", map_location=torch.device('cpu'))

real_data_file = './data/porto_2w_iter120_b4.txt'
syn_data_file = './data/two_dim_porto_2w_iter120_b4_1.txt'

real_start_loc, real_start_time = get_real_data(real_data_file)

syn_start_loc = get_all_syn_data(syn_data_file)

syn_start_loc_reshaped = syn_start_loc.reshape(-1, syn_start_loc.shape[-1])
real_start_time_reshaped = real_start_time.reshape(-1, real_start_time.shape[-1])

mm_syn_start_loc = MinMaxScaler()
mm_real_start_time = MinMaxScaler()

syn_start_loc_m = mm_syn_start_loc.fit_transform(syn_start_loc_reshaped).reshape(syn_start_loc.shape)
real_start_time_m = mm_real_start_time.fit_transform(real_start_time_reshaped).reshape(real_start_time.shape)
input_dataset = DataLoader(syn_start_loc_m, shuffle=False, batch_size=1)
# inputs = torch.tensor(syn_start_loc_m).to(torch.float32).to(device)

load_model.eval()

# filename = os.path.join('./gen_start_time', "gen_top5_time1.txt")
trajs_with_time = []
count = 0

for index, data in enumerate(input_dataset):
    inputs = data.to(device).to(torch.float32)
    noise = torch.randn(1, 5, 1).to(device)
    output = load_model(inputs[:, :5, :], noise).to(torch.float32)
    numpy_array = output.cpu().detach().numpy()
    numpy_array_reshaped = numpy_array.reshape(-1, real_start_time.shape[-1])
    gen_time_new = mm_real_start_time.inverse_transform(numpy_array_reshaped)
    each_traj = []  # 在每次迭代开始时清空
    # 将 inputs 还原
    inputs_re = mm_syn_start_loc.inverse_transform(inputs.squeeze(0).cpu().numpy())
    t = gen_time_new[0][0]  # 第一个时间
    for i in range(inputs.shape[1]):
        lng, lat = inputs_re[i]
        if lng != 0 and lat != 0:
            each_traj.append([lng, lat, t])
    trajs_with_time.append(each_traj)
    count += 1
    print("添加了", count, '条轨迹！')

# print(len(trajs_with_time))

txt_path = './gen_start_time/6w_b4_120s_LTSGAN/time_and_loc_120s_b5_2w_zkf_5_change.txt'
formalize_data(trajs_with_time, txt_path, is_timeinterval=False, without_time=False)

print(100 * '-')
print('开始以时间间隔递增！')
out_path = r'D:/Learning sources/python project/TS-TrajGAN-main/metrics_data/2w_b5_120s_LTSGAN/5_change.txt'
change_time(txt_path, 120, out_path)
