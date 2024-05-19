from cgan import train

# 设置超参数
input_size = 3  # 输入维度，假设是轨迹的坐标
hidden_size = 128  # 隐藏层维度
output_size = 1  # 输出维度
lr = 0.001  # 学习率
num_epochs = 1000
batch_size = 64
seq_len = 5

real_file = './data/porto_6w_iter120_b4.txt'
syn_file = './data/two_dim_porto_2w_iter120_b4_1.txt'
# 训练模型
train(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len,
      batch_size=batch_size, lr=lr, real_data_file=real_file, syn_data_file=syn_file,
      num_epochs=num_epochs)
