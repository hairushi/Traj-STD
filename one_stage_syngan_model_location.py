# Necessary Packages
# import warnings
# warnings.filterwarnings("ignore")
import math
import os
import pathlib

from matplotlib import pyplot as plt
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import numpy as np
from utils import extract_time, rnn_cell, random_generator, padding, draw_loss_pic
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, RobustScaler, MinMaxScaler
from location_load import read_txt_traj_data_loading

def syngan(ori_data, parameters):
    #将原始轨迹数据映射到潜在空间（latent space）。
    #这个嵌入过程有两个主要阶段：编码（Embedding）和解码（Recovery）。
    """syngan function.

        Use original initial trajectory segments as training set to generate initial trajectory segments

        Args:
            - ori_data: original initial trajectory segments
            - parameters: syngan network parameters

        Returns:
            - generated_data: generated initial trajectory segments
        """

    # Initialization on the Graph
    tf.reset_default_graph()#清除当前默认图形，以便重新构建新的计算图

    # utils
    def batch_generator(data, time, batch_size):
        """Mini-batch generator.

        Args:
          - data: initial trajectory segments
          - time: time information
          - batch_size: the number of samples in each batch

        Returns:
          - X_mb: initial trajectory segments in each batch
          - T_mb: time information in each batch
        """
        no = len(data)
        idx = np.random.permutation(no)
        train_idx = idx[:batch_size]

        # 包含了当前 mini-batch 中每个轨迹的前 max_seq_len - 1 个时间步的数据
        X_mb = list(data[i][:-1] for i in train_idx)
        #取 data[i] 中最后一个时间步的数据，然后从这个数据中取第一个元素，data[i][-1] 包含了轨迹的长度信息。
        Length_mb = list(data[i][-1][0] for i in train_idx)
        T_mb = list(time[i] for i in train_idx)
        Length_mb = np.expand_dims(Length_mb, axis=1)#为了符合网络输入的情况

        return X_mb, Length_mb, T_mb

    ## Build a RNN networks
    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    out_data_path = parameters['out_data_path']

    # Maximum sequence length and each sequence length
    max_seq_len = parameters['cut_seq_len'] + 1
    dim = len(ori_data[0][0])
    z_dim = dim

    ori_time_list = np.array(extract_time(ori_data)[0]) - 1
    ##每条轨迹长度-1

    syn_time_list = ori_time_list[:]

    data_for_norm = list()
    for each_traj in ori_data:
        for each_coor in each_traj[:-1]:
            data_for_norm.append(np.array(each_coor))
    #MinMaxScaler 将每个特征缩放到给定的范围（默认是 [0, 1]）。
    # fit_transform 方法用于计算并应用归一化的转换，最终得到 data_norm。
    stand_scaler = MinMaxScaler()
    stand_scaler.fit_transform(data_for_norm)
    data_norm = stand_scaler.transform(data_for_norm)

    norm_data_index = 0
    for i in range(len(ori_time_list)):
        ori_data[i][:-1] = data_norm[norm_data_index: norm_data_index + int(ori_time_list[i])]
        norm_data_index += int(ori_time_list[i])
    #将归一化后的数据 data_norm 分配回原始数据 ori_data 中（不包括将归一化的轨迹长度信息赋值）。
    #具体而言，它遍历了 ori_time_list 中的每个元素，该列表包含每个轨迹的长度。
    #然后，通过 norm_data_index 来控制从 data_norm 中截取相应长度的数据，
    #并将这部分数据赋值给对应轨迹的前 len - 1 个坐标点（排除最后一个用于表示长度的元素）。

    length_data_for_norm = list()#取每条轨迹的归一化后的长度
    for each_traj in ori_data:
        length_data_for_norm.append(np.array(each_traj[-1]))
    stand_scaler_length = MinMaxScaler()
    stand_scaler_length.fit_transform(length_data_for_norm)
    length_data_norm = stand_scaler_length.transform(length_data_for_norm)
    for i in range(len(ori_data)):
        ori_data[i][-1] = length_data_norm[i]
    #对轨迹数据的长度进行了归一化操作。
    # 首先，遍历原始轨迹数据 (ori_data) 中的每个轨迹 (each_traj)，
    # 提取每个轨迹的长度信息 (each_traj[-1])，并将其存储在 length_data_for_norm 列表中。
    # 接着，使用 MinMaxScaler 对长度数据进行归一化，
    # 然后将归一化后的长度数据重新赋值给原始轨迹数据的最后一个元素 (ori_data[i][-1])。

    # data input for Predictor
    train_data_for_predictor = list()
    for each_traj in ori_data:#训练时只把轨迹中位置点个数大于begin_seq_len的轨迹纳入到训练数据中
        if np.isnan(each_traj[-1][0]):
            pass
        else:
            train_data_for_predictor.append(each_traj)

    # Input placeholders
    X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")#其中 None 表示第一维可以是任意长度，max_seq_len 表示第二维的长度，dim 表示第三维的长度。这个占位符通常用于表示输入的轨迹数据，其中第一维是轨迹数量，第二维是时间步数，第三维是特征维度。
    Length = tf.placeholder(tf.float32, [None, 1], name="goal_length")
    Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.placeholder(tf.int32, [None], name="myinput_t")
    #X：代表输入的轨迹数据，是一个三维的张量，形状为 [None, max_seq_len, dim]。
    #Length：代表轨迹的长度信息，是一个二维的张量，形状为 [None, 1].
    #Z：代表随机生成的向量，用于生成轨迹的隐藏表示，也是一个三维的张量，形状为 [None, max_seq_len, z_dim]。
    #T：代表轨迹的时间信息，是一个一维的整数张量，形状为 [None]
    #这段代码是一个典型的 TensorFlow 训练数据输入的设置，确保了输入数据的维度和类型符合网络的期望

    def embedder(X, T):
        """Embedding network between original feature space to latent space.

        Args:
          - X: input initial trajectory segments features
          - T: input time information

        Returns:
          - H: embeddings
        """
        with tf.variable_scope("embedder", reuse=tf.AUTO_REUSE):
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])#用于创建 RNN 单元
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)#接受输入 X 和时间信息 T，并返回 RNN 网络的输出 e_outputs 和最终状态 e_last_states。
            H = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='dense')#使用全连接层将 RNN 网络的输出 e_outputs 映射到潜在空间 H。
        return H

    def recovery(H, T):
        #通过学习将潜在表示 H 转换回原始轨迹片段，从而使生成的合成轨迹数据更好地匹配原始数据的分布。
        """recovery network from latent space to original space.

        Args:
          - H: latent representationlatent representation
          "（潜在表示）通常指的是在学习过程中模型学到的数据的低维表示.
          - T: input time information

        Returns:
          - X_tilde: recovered initial trajectory segments
        """
        with tf.variable_scope("recovery", reuse=tf.AUTO_REUSE):
            de_cell = tf.keras.layers.StackedRNNCells(
                [rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])  # module_name = 'gru'
            de_outputs, de_last_states = tf.nn.dynamic_rnn(de_cell, H, dtype=tf.float32, sequence_length=T)
            X_tilde = tf.layers.dense(de_outputs, dim, activation=tf.nn.sigmoid, name='dense')
        return X_tilde#函数返回了重构后的轨迹数据 X_tilde，它表示通过潜在空间映射回原始空间后得到的轨迹数据。

    def predictor(H, T):
        """Predict the initial trajectory's length from latent space.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - Pred_length: Predicted length results
        """
        with tf.variable_scope("predictor", reuse=tf.AUTO_REUSE):
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
            pooled = tf.reduce_mean(e_outputs, axis=1)
            Pred_length = tf.layers.dense(pooled, 1, activation=tf.nn.sigmoid, name='dense')
        return Pred_length

    def generator(Z, T):
        """Generator function: Generate initial trajectory segments in latent space.

        Args:
          - Z: random variables
          - T: input time information

        Returns:
          - E: generated embedding
        """
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length=T)
            E = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='dense')
        return E

    def supervisor(H, T):
        """Generate next sequence using the previous sequence.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - S: generated sequence based on the latent representations generated by the generator
        """
        with tf.variable_scope("supervisor", reuse=tf.AUTO_REUSE):
            e_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers - 1)])
            e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length=T)
            S = tf.layers.dense(e_outputs, hidden_dim, activation=tf.nn.sigmoid, name='dense')
        return S

    def discriminator(H, T):
        """Discriminate the original and synthetic initial trajectory segments.

        Args:
          - H: latent representation
          - T: input time information

        Returns:
          - Y_hat: classification results between original and synthetic initial trajectory segments
        """
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            d_cell = tf.keras.layers.StackedRNNCells([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
            d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length=T)
            Y_hat = tf.layers.dense(d_outputs, 1, activation=None, name='dense')
        return Y_hat

    # embedder & recovery
    H = embedder(X, T)  # (?, seq_len, hidden_dim)
    X_tilde = recovery(H, T)  # (?, seq_len, ori_dim)

    # Predictor
    pred_length = predictor(H, T)

    # Generator
    E_hat = generator(Z, T)  # （?, seq_len, hidden_dim）
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)

    # Synthetic data
    X_hat = recovery(H_hat, T)

    # Discriminator
    Y_real = discriminator(H, T)
    Y_fake = discriminator(H_hat, T)
    Y_fake_e = discriminator(E_hat, T)

    # Variables
    e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
    r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
    pred_vars = [v for v in tf.trainable_variables() if v.name.startswith('predictor')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

    gamma = 1

    # Predictor loss
    Pred_loss = tf.losses.mean_squared_error(pred_length, Length)

    # Discriminator loss
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
    D_loss_adv = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    D_loss = D_loss_adv

    # Generator loss
    # 1. Adversarial loss
    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)

    # 2. Supervised loss
    G_loss_S = tf.losses.mean_squared_error(H[:, :, :], H_hat_supervise[:, :, :])

    # 2. Two Moments
    G_loss_V1 = tf.reduce_mean(
        tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0])))
    G_loss_V = G_loss_V1 + G_loss_V2

    # 3. Summation
    G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100 * G_loss_V

    # embedder network loss
    E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
    E_loss0 = 10 * tf.sqrt(E_loss_T0)
    E_loss = E_loss0 + 0.1 * G_loss_S

    # optimizer
    learning_rate = 0.001  # 设置你想要的学习率
    E0_solver = tf.train.AdamOptimizer(learning_rate).minimize(E_loss0, var_list=e_vars + r_vars)
    GS_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss_S, var_list=g_vars + s_vars)
    Pred_solver = tf.train.AdamOptimizer(learning_rate).minimize(Pred_loss, var_list=pred_vars)
    E_solver = tf.train.AdamOptimizer(learning_rate).minimize(E_loss, var_list=e_vars + r_vars)
    D_solver = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars + s_vars)

    ## syngan training
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver_en_de = tf.train.Saver(var_list=e_vars + r_vars)
    saver_sup = tf.train.Saver(var_list=g_vars + s_vars + pred_vars)
    saver_joint = tf.train.Saver(var_list=e_vars + r_vars + d_vars + g_vars + s_vars + pred_vars)

    save_model_para_path = out_data_path + "/One_Stage_Model/"
    save_en_de_para_path = save_model_para_path + "Em_recovery_Model/"
    save_sup_pred_para_path = save_model_para_path + "Sup_Model/"
    save_gan_joint_para_path = save_model_para_path + "GAN_joint_Model/"
    pathlib.Path(save_model_para_path).mkdir(parents=True, exist_ok=True)

    losses_only = {'embedder_loss': [], 'sup_loss': [], 'pred_loss': []}
    losses_joint = {'embedder_loss': [], 'g_loss': [], 'd_loss': [], 'pred_loss': []}
    save_path = out_data_path + 'loss_pic/one_stage/'
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    print('Traing Parameters:')
    print('batch_size = ' + str(batch_size))
    print('hidden_dim = ' + str(hidden_dim))
    print('num_layers = ' + str(num_layers))
    print('module_name = ' + str(module_name))
    print('Start embedder & recovery Network Training')
    print_iter = 1000
    num_epochs = int(iterations / print_iter)

    if os.path.exists(save_en_de_para_path + 'checkpoint'):
        print("Loading the parameters of the embedder and recovery has already trained in one stage...")
        saver_en_de.restore(sess, save_en_de_para_path + 'en_de_model.ckpt')
    else:
        pathlib.Path(save_en_de_para_path).mkdir(parents=True, exist_ok=True)
        print('Start Embedding Network Training in one stage...')
        # 1. Embedding network training(embedder & recovery)
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                X_mb, _, T_mb = batch_generator(ori_data, ori_time_list, batch_size)
                # Train embedder
                _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: padding(X_mb, max_seq_len), T: T_mb})
                # Checkpoint
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(e_loss=np.round(np.sqrt(step_e_loss), 4))
                losses_only['embedder_loss'].append(np.round(np.sqrt(step_e_loss), 4))
        draw_loss_pic(losses_only['embedder_loss'], save_path + 'one_stage_' + 'embedder_only_loss.png')
        save_path1 = os.path.join(save_en_de_para_path, "en_de_model.ckpt")
        os.chmod(save_en_de_para_path, 0o755)
        saver_en_de.save(sess, save_path1)
        print('Finish embedder & recovery Network Training')

    if os.path.exists(save_sup_pred_para_path + 'checkpoint'):
        print("Loading the parameters of the supervisor and predictor has already trained in one stage...")
        saver_sup.restore(sess, save_sup_pred_para_path + 'sup_model.ckpt')
    else:
        pathlib.Path(save_sup_pred_para_path).mkdir(parents=True, exist_ok=True)
        # 2. Training with supervised loss and prediction loss
        print('Start Training Supervisor and Predictor Network Only in one stage...')
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                # Set mini-batch
                X_mb, _, T_mb = batch_generator(ori_data, ori_time_list, batch_size)

                X_mb_for_pre, Length_mb_for_pre, T_mb_for_pre = \
                    batch_generator(train_data_for_predictor, ori_time_list, batch_size)
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                # Train generator
                _, step_g_loss_s = sess.run([GS_solver, G_loss_S],
                                            feed_dict={Z: padding(Z_mb, max_seq_len), X: padding(X_mb, max_seq_len),
                                                       T: T_mb})
                _, step_pred_loss = sess.run([Pred_solver, Pred_loss],#！！
                                             feed_dict={X: X_mb_for_pre, Length: Length_mb_for_pre, T: T_mb_for_pre})
                # Checkpoint
                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(s_loss=np.round(np.sqrt(step_g_loss_s), 4),
                                 pred_loss=np.round(np.sqrt(step_pred_loss), 4))
                losses_only['sup_loss'].append(np.round(np.sqrt(step_g_loss_s), 4))
                losses_only['pred_loss'].append(np.round(np.sqrt(step_pred_loss), 4))
        draw_loss_pic(losses_only['sup_loss'], save_path + 'one_stage_' + 'sup_only_loss.png')
        draw_loss_pic(losses_only['pred_loss'], save_path + 'one_stage_' + 'pred_only_loss.png')
        save_path2 = os.path.join(save_sup_pred_para_path, "sup_model.ckpt")
        os.chmod(save_sup_pred_para_path, 0o755)
        saver_sup.save(sess, save_path2)
        print('Finish Training Supervisor and Predictor Network Only')

    if os.path.exists(save_gan_joint_para_path + 'checkpoint'):
        print("Loading the parameters of joint training model has already trained in one stage...")
        saver_joint.restore(sess, save_gan_joint_para_path + 'joint_model.ckpt')
    else:
        pathlib.Path(save_gan_joint_para_path).mkdir(parents=True, exist_ok=True)
        # 3. Joint Training(5models)
        print('Start Joint Training in one stage...')
        for epoch in range(num_epochs):
            loop = tqdm(range(print_iter))
            for _ in loop:
                # Set mini-batch
                for _ in range(2):
                    X_mb, _, T_mb = batch_generator(ori_data, ori_time_list, batch_size)
                    # Random vector generation
                    Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                    # Train generator
                    _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V],
                                                                              feed_dict={Z: padding(Z_mb, max_seq_len),
                                                                                         X: padding(X_mb, max_seq_len),
                                                                                         T: T_mb})
                    # Train embedder
                    _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: padding(Z_mb, max_seq_len),
                                                                                   X: padding(X_mb, max_seq_len),
                                                                                   T: T_mb})
                # Discriminator training
                # Set mini-batch
                X_mb, _, T_mb = batch_generator(ori_data, ori_time_list, batch_size)
                X_mb_for_pre, Length_mb_for_pre, T_mb_for_pre = \
                    batch_generator(train_data_for_predictor, ori_time_list, batch_size)
                # Random vector generation
                Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
                # Train discriminator
                _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: padding(X_mb, max_seq_len), T: T_mb,
                                                                         Z: padding(Z_mb, max_seq_len)})
                _, step_pred_loss = sess.run([Pred_solver, Pred_loss],
                                             feed_dict={X: X_mb_for_pre, Length: Length_mb_for_pre, T: T_mb_for_pre})

                loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                loop.set_postfix(d_loss=np.round(step_d_loss, 4),
                                 g_loss=np.round(step_g_loss_u + step_g_loss_v, 4),
                                 g_loss_s=np.round(np.sqrt(step_g_loss_s), 4),
                                 e_loss_t=np.round(np.sqrt(step_e_loss_t0), 4),
                                 pred_loss=np.round(np.sqrt(step_pred_loss), 4))
                losses_joint['g_loss'].append(np.round(step_g_loss_u + step_g_loss_v, 4))
                losses_joint['d_loss'].append(np.round(step_d_loss, 4))
                losses_joint['embedder_loss'].append(np.round(np.sqrt(step_e_loss_t0), 4))
                losses_joint['pred_loss'].append(np.round(np.sqrt(step_pred_loss), 4))
        draw_loss_pic(losses_joint['embedder_loss'], save_path + 'one_stage_' + 'embedder_joint_loss.png')
        draw_loss_pic(losses_joint['pred_loss'], save_path + 'one_stage_' + 'pred_joint_loss.png')
        plt.figure(figsize=(8, 6), dpi=100)
        plt.plot(range(len(losses_joint['g_loss'])), losses_joint['g_loss'], color='r', label='Generator')
        plt.plot(range(len(losses_joint['d_loss'])), losses_joint['d_loss'], color='b', label='Discriminator')
        plt.xlabel('Time')
        plt.ylabel('Loss')
        plt.title('GAN Training Loss')
        plt.legend()
        plt.savefig(save_path + 'one_stage_' + 'G&D_joint_loss.png', format='png')
        plt.close()
        print('Finish Generator and discriminator Training')
        save_path3 = os.path.join(save_gan_joint_para_path, "joint_model.ckpt")
        os.chmod(save_gan_joint_para_path, 0o755)
        saver_joint.save(sess, save_path3)

    print("Start Initial Trajectory Segments Generating...")

    ## Synthetic data generation
    pred_traj_num = len(syn_time_list)
    Z_mb = random_generator(pred_traj_num, z_dim, syn_time_list, max_seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: padding(Z_mb, max_seq_len), T: syn_time_list})

    to_end_traj_data = list()
    to_two_stage_traj_data = list()
    to_two_stage_traj_length = list()

    for i in tqdm(range(pred_traj_num)):
        each_gen = generated_data_curr[i, :syn_time_list[i], :]
        each_gen_inverse = stand_scaler.inverse_transform(each_gen)
        if len(each_gen) < max_seq_len:
            to_end_traj_data.append(each_gen_inverse)
        else:
            pred_length_curr = sess.run(pred_length, feed_dict={X: np.array([each_gen]), T: np.array([max_seq_len])})
            pred_length_res = \
                np.squeeze(
                    stand_scaler_length.inverse_transform([np.stack([np.squeeze(pred_length_curr)] * 2, axis=0)]))[0]
            if math.floor(pred_length_res) < max_seq_len + 1:
                to_end_traj_data.append(each_gen_inverse)
            else:
                to_two_stage_traj_data.append(each_gen_inverse)
                to_two_stage_traj_length.append(pred_length_res)

    return to_end_traj_data, to_two_stage_traj_data, to_two_stage_traj_length


# path='./data/porto_grid20_b4_2w.txt'
#
# one_stage_train_data, two_stage_train_data, _ = \
#         read_txt_traj_data_loading(path)
#
#
# parameters = {
#     'hidden_dim': 64,
#     'num_layers': 2,
#     'iterations': 10000,
#     'batch_size': 64,
#     'module': 'gru',  # 或者 'lstm'，根据您的需求选择
#     'cut_seq_len': 100,  # 裁剪的序列长度
#     'out_data_path': './output_onestage',
#     'num_layer':3# 输出数据路径
#     # 添加更多参数...
# }

# # 调用 syngan 函数
# to_end_traj_data, to_two_stage_traj_data, to_two_stage_traj_length = syngan(one_stage_train_data, parameters)
#
# # 输出结果
# print("生成的初始轨迹段（结束于第一阶段）：")
# for traj in to_end_traj_data:
#     print(traj)
#
# print("\n生成的初始轨迹段（传递到第二阶段的）：")
# for traj in to_two_stage_traj_data:
#     print(traj)
#
# print("\n传递到第二阶段的轨迹段的长度：")
# for length in to_two_stage_traj_length:
#     print(length)