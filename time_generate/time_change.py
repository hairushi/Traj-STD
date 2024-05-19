import os
import time


def get_float_time(timestamp, begin_float_time):
    if len(timestamp) == 19:
        return time.mktime(time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')) - begin_float_time
    else:
        return time.mktime(time.strptime(timestamp[:19], '%Y-%m-%d %H:%M:%S')) - begin_float_time


def change_time(txt_path, time_interval, out_path):
    with open(txt_path, 'r') as f:
        con = f.readlines()  # 读取所有行数据

    data_list = []  # 存储所有轨迹数据
    current_traj = []  # 当前轨迹的数据
    count = 0
    for line in con:
        if line.startswith('#'):  # 遇到新轨迹的起始位置
            if current_traj:  # 如果当前轨迹数据不为空，则将其添加到轨迹数据列表中
                data_list.append(current_traj)
            current_traj = []  # 重置当前轨迹数据
        else:
            traj_info = line.strip('>0:').strip('\n').split(';')[:-1]  # 提取轨迹信息
            each_traj = []
            traj_begin_str_time = traj_info[0].split(',')[2]  # 读取轨迹起始时间
            traj_begin_float_time = time.mktime(time.strptime(traj_begin_str_time, '%Y-%m-%d %H:%M:%S'))
            for point in traj_info:
                lng, lat, timestamp = point.split(',')
                time_info = traj_begin_float_time + count * time_interval  # 计算时间信息
                each_traj.append((float(lng), float(lat), float(time_info)))
                count = count + 1
            current_traj.append(each_traj)  # 将当前轨迹数据添加到当前轨迹列表中

    # 处理最后一条轨迹数据
    if current_traj:
        data_list.append(current_traj)

    # 创建输出文件夹（如果不存在）
    out_folder = os.path.dirname(out_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(out_path, 'w') as f:
        for i, traj in enumerate(data_list):
            print("成功写入第", i, "条轨迹！")
            f.write("#" + str(i) + ":" + "\r")
            for each_traj in traj:
                f.write(">0:")
                for point in each_traj:
                    lng = str(round(point[0], 6))
                    lat = str(round(point[1], 6))
                    time_info = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(point[2]))
                    f.write(lng + "," + lat + "," + time_info + ";")
                f.write("\n")  # 添加换行符

    print('Write success!')

# def change_time1(txt_path, time_interval, out_path,ori_path):
#     with open(txt_path, 'r') as f:
#         con = f.readlines()  # 读取所有行数据
#
#     with open(ori_path, 'r') as f:
#         ori = f.readlines()  # 读取所有行数据
#
#     time_list=[]
#     for line in ori:
#         if line.startswith('>0:'):  # 遇到新轨迹的起始位置
#             t= line.strip('>0:').strip('\n').split(';')[:-1]
#             traj_begin_str_time = t[0].split(',')[2]  # 读取轨迹起始时间
#             #print(traj_begin_str_time)
#             traj_begin_float_time = time.mktime(time.strptime(traj_begin_str_time, '%Y-%m-%d %H:%M:%S'))
#             time_list.append(traj_begin_float_time)
#
#     data_list = []  # 存储所有轨迹数据
#     current_traj = []  # 当前轨迹的数据
#     count_traj=0
#     for line in con:
#         if line.startswith('#'):  # 遇到新轨迹的起始位置
#             if current_traj:  # 如果当前轨迹数据不为空，则将其添加到轨迹数据列表中
#                 data_list.append(current_traj)
#             current_traj = []  # 重置当前轨迹数据
#         else:
#             traj_info = line.strip('>0:').strip('\n').split(';')[:-1]  # 提取轨迹信息
#             each_traj = []
#             count = 0
#             for point in traj_info:
#                 lng, lat, timestamp = point.split(',')
#                 time_info = time_list[count_traj]+ count * time_interval  # 计算时间信息
#                 each_traj.append((float(lng), float(lat), float(time_info)))
#                 count = count + 1
#             current_traj.append(each_traj)# 将当前轨迹数据添加到当前轨迹列表中
#             count_traj=count_traj+1

#
# # 处理最后一条轨迹数据
# if current_traj:
#     data_list.append(current_traj)
#
# # 创建输出文件夹（如果不存在）
# out_folder = os.path.dirname(out_path)
# if not os.path.exists(out_folder):
#     os.makedirs(out_folder)
#
# with open(out_path, 'w') as f:
#     for i, traj in enumerate(data_list):
#         print("成功写入第", i, "条轨迹！")
#         f.write("#" + str(i) + ":" + "\r")
#         for each_traj in traj:
#             f.write(">0:")
#             for point in each_traj:
#                 lng = str(round(point[0], 6))
#                 lat = str(round(point[1], 6))
#                 time_info = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(point[2]))
#                 f.write(lng + "," + lat + "," + time_info + ";")
#             f.write("\n")  # 添加换行符
#
# print('Write success!')

# txt_path = r'D:/Learning sources/python project/TS-TrajGAN-main/porto_6w_iter30_b4_first_time/syn_9_pred_1/syn_data_output/gen_cgan_first_time_6w.txt'
# out_path = r'D:/Learning sources/python project/TS-TrajGAN-main/porto_6w_iter30_b4_first_time/syn_9_pred_1/syn_data_output/cgen_final_combine_30s.txt'
# #ori=r'./data/porto_6w_iter30_b4.txt'
# #change_time1(txt_path, 30, out_path,ori)
# change_time(txt_path,120,out_path)
# txt_path='./gen_start_time/time_and_loc_120s_b4_6w_4.txt'
# # formalize_data(trajs_with_time, txt_path, is_timeinterval=False, without_time=False)
#
# print(100*'-')
# print('开始以时间间隔递增！')
# out_path = r'D:/Learning sources/python project/TS-TrajGAN-main/metrics_data/6w_b4_120s/4.txt'
# change_time(txt_path,120,out_path)
