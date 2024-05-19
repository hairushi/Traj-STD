## Necessary Packages
import math
import numpy as np
from utils import padding


def read_txt_traj_data_loading(txt_path, begin_seq_len=5, pred_seq_len=1, max_seq_len=50,
                               boundary=(41.06421, 41.20999, -8.661858, -8.525016), with_end_sign=False,
                               for_test=False):
    end_sign = list([boundary[0] - (boundary[1] - boundary[0]), boundary[2] - (boundary[-1] - boundary[-2]), -1])
    with open(txt_path, 'r') as f:
        con = f.readlines()[:]  # readlines() 方法用于读取所有行，而 [:] 则是 Python 中的切片语法，表示将整个列表复制一份。
    one_stage_read_begin_traj = list()
    two_stage_read_cut_traj = list()
    metrics_traj = list()
    # time_interval_list = list()

    for i in range(int(len(con) / 2)):
        traj = con[2 * i + 1].strip('>0:').strip('\n').split(';')[:-1]
        each_traj = list()
        for i in range(len(traj)):
            traj[i] = traj[i].split(',')
            if for_test:
                coor = (float(traj[i][1]), float(traj[i][0]))
            else:
                coor = (float(traj[i][0]), float(traj[i][1]))
            each_traj.append(coor)

        metrics_traj.append(each_traj[:max_seq_len])  # 读取每条轨迹前max_seq_len位置点
        # the train set of one stage
        if len(each_traj) < begin_seq_len:
            each_length = math.nan  # 将小于设置训练的初始位置点个数的轨迹长度设置为nan
            temp_list = each_traj
        else:
            each_length = len(each_traj)
            temp_list = each_traj[:begin_seq_len]
        temp_list.append((each_length, each_length))
        one_stage_read_begin_traj.append(temp_list)

        # the train set of two stage
        cut_seq_len = begin_seq_len - 1 + pred_seq_len
        each_for_cut = each_traj[1:]
        if len(each_for_cut) < cut_seq_len:
            pass
        else:
            # each_for_cut = np.array(each_for_cut)
            # for j in range(len(each_for_cut) - cut_seq_len + 1):
            #     cut_list = each_for_cut[j: j + cut_seq_len, :]
            #     two_stage_read_cut_traj.append(cut_list)
            if with_end_sign:
                each_for_cut = np.array(each_for_cut)
                new_cut = np.insert(each_for_cut, len(each_for_cut), end_sign, axis=0)
            else:
                new_cut = np.array(each_for_cut)
            for j in range(len(new_cut) - cut_seq_len + 1):
                cut_list = new_cut[j: j + cut_seq_len, :]
                two_stage_read_cut_traj.append(cut_list)

    one_stage_read_begin_traj = one_stage_read_begin_traj[::-1]  # 末尾开始，逆向取出所有元素
    idx = np.random.permutation(len(one_stage_read_begin_traj))
    one_stage_data = []
    for i in range(len(one_stage_read_begin_traj)):
        one_stage_data.append(one_stage_read_begin_traj[idx[
            i]])  # 将经过打乱顺序后的 one_stage_read_begin_traj 中的元素按照随机排列的索引 idx 的顺序逐个添加到 one_stage_data 列表中。

    #    print("the first GAN 's train dataset's shape is " + str(np.array(one_stage_data).shape[0]))

    # shuffle data
    two_stage_read_cut_traj = two_stage_read_cut_traj[::-1]
    idx = np.random.permutation(len(two_stage_read_cut_traj))
    two_stage_data = []
    for i in range(len(two_stage_read_cut_traj)):
        two_stage_data.append(two_stage_read_cut_traj[idx[i]])

    # print("the second CGAN 's train dataset's shape is" + str(np.array(two_stage_data).shape))

    metrics_traj = metrics_traj[::-1]
    # #    print("the metrics data's shape is " + str(np.array(metrics_traj).shape[0]))
    #     print("Metrics traj:")
    #     for traj in metrics_traj:
    #         print(traj)
    metrics_traj_padded = padding(metrics_traj, max_seq_len)
    return one_stage_data, two_stage_data, metrics_traj_padded


# ori_traj_txt_path = './data/porto_grid20_b4_2w.txt'
# # syn_traj_txt_path = './data/formalized_txt_syn9_pred1_min1_max30_with_time.txt'
# #
# ori_data = np.array(read_txt_traj_data_loading(ori_traj_txt_path, max_seq_len=30, for_test=True)[-1])
# print(ori_data)

if __name__ == '__main__':
    ori_traj_txt_path = './data/porto_2w_iter120_b5.txt'
    read_txt_traj_data_loading(ori_traj_txt_path)
