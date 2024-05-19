from datetime import datetime
import numpy as np


def get_real_data(file):
    start_loc = []
    start_time = []
    each_top_five_loc = []
    each_top_five_time = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if i % 2 == 0:
            continue
        else:
            for j in range(5):
                if j == 0:
                    segments = lines[i].strip().split(';')
                    first_segment = segments[0].split('>0:')[1]
                    first_coordinate = first_segment.split(',')[0:2]
                    first_time_str = first_segment.split(',')[2]
                    first_time = datetime.strptime(first_time_str, '%Y-%m-%d %H:%M:%S')
                    t = (first_time - first_time.replace(hour=0, minute=0, second=0)).total_seconds()
                    latitude = float(first_coordinate[0])
                    longitude = float(first_coordinate[1])
                    each_top_five_loc.append([latitude, longitude])
                    each_top_five_time.append(t)
                else:
                    coordinate = segments[j].split(',')[0:2]
                    time_str = segments[j].split(',')[2]
                    time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                    t = (time - time.replace(hour=0, minute=0, second=0)).total_seconds()
                    latitude = float(coordinate[0])
                    longitude = float(coordinate[1])
                    each_top_five_loc.append([latitude, longitude])
                    each_top_five_time.append(t)
            start_loc.append(each_top_five_loc)
            start_time.append(each_top_five_time)
            each_top_five_loc = []
            each_top_five_time = []
    start_loc = np.array(start_loc)
    start_time = np.array(start_time)
    start_time.resize(len(start_time), 5, 1)
    return start_loc, start_time


def get_syn_data(file):
    start_loc = []
    each_top_five_loc = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if i % 2 == 0:
            continue
        else:
            for j in range(5):
                if j == 0:
                    segments = lines[i].strip().split(';')
                    first_segment = segments[0].split('>0:')[1]
                    first_coordinate = first_segment.split(',')[0:2]
                    latitude = float(first_coordinate[0])
                    longitude = float(first_coordinate[1])
                    each_top_five_loc.append([latitude, longitude])
                else:
                    coordinate = segments[j].split(',')[0:2]
                    latitude = float(coordinate[0])
                    longitude = float(coordinate[1])
                    each_top_five_loc.append([latitude, longitude])
            start_loc.append(each_top_five_loc)
            each_top_five_loc = []
    start_loc = np.array(start_loc)
    return start_loc


import numpy as np

def get_all_syn_data(file):
    with open(file, 'r') as f:
        con = f.readlines()  # 读取所有行数据

    data_list = []  # 存储所有轨迹数据
    current_traj = []  # 当前轨迹的数据
    for line in con:
        if line.startswith('#'):  # 遇到新轨迹的起始位置
            if current_traj:  # 如果当前轨迹数据不为空，则将其添加到轨迹数据列表中
                data_list.extend(current_traj)
            current_traj = []  # 重置当前轨迹数据
        else:
            traj_info = line.strip('>0:').strip('\n').split(';')[:-1]  # 提取轨迹信息
            each_traj = []
            for point in traj_info:
                lng, lat = point.split(',')
                each_traj.append([float(lng), float(lat)])
            current_traj.append(each_traj)  # 将当前轨迹数据添加到当前轨迹列表中

    # 处理最后一条轨迹数据
    if current_traj:
        data_list.extend(current_traj)

    max_length = max(len(traj) for traj in data_list)

    for i in range(len(data_list)):
        while len(data_list[i]) < max_length:
            data_list[i].append([0.0, 0.0])

    array=np.array(data_list)

    return array

if __name__ == '__main__':
    data_list = get_syn_data('./data/two_dim_porto_2w_iter120_b4_1.txt')
#print(get_all_syn_data())
#print(len(get_all_syn_data()))
#print(get_all_syn_data().shape)
# all_syn_data = get_all_syn_data()
# five_syn_data = []
# for traj in all_syn_data:
#     five_syn_data.append(traj[:5])
#
# start_loc=np.array(five_syn_data)
# print(start_loc.shape)



