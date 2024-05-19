import ast
import csv
import random
import time
import numpy as np


def checkCellIndex(each_coor):  # coor是个1*2数组，boundary是个1*4数组

    if boundary[0] < float(each_coor[0]) < boundary[1] and boundary[2] < float(each_coor[1]) < boundary[3]:
        height_cell = (boundary[1] - boundary[0]) / grid_size
        width_cell = (boundary[3] - boundary[2]) / grid_size

        cloumnindex = int((float(each_coor[1]) - boundary[2]) / width_cell)
        rowindex = int((boundary[1] - float(each_coor[0])) / height_cell)

        return (rowindex, cloumnindex)
    else:
        return False


def data_extract():
    with open('./data/train.csv') as csvfile:
        rows = csvfile.readlines()[1:]
        num = int(len(rows) / 2)
        traj_count_id = 0
        whole_count = 0
        final_length_list = []
    with open(out_path, 'w') as f:
        for each_traj in rows[:num]:
            each_traj = ast.literal_eval(each_traj)
            if each_traj[7] == 'False':
                tj1 = each_traj[8]
                if tj1 != '[]':
                    tj_begin_time = int(each_traj[5])
                    tj = np.array([ast.literal_eval(tj1)])
                    tj = tj.reshape(tj.shape[1], 2)
                    time_iterval = 60
                    flag = True
                    coor_list = []
                    for i in range(tj.shape[0]):
                        coor = (tj[i][0], tj[i][1])
                        cell_index = checkCellIndex(coor)
                        if cell_index:
                            if i == 0:
                                time_info = tj_begin_time
                            else:
                                time_info = time_iterval * i + tj_begin_time
                            time_info_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_info))
                            coor_info = (coor[1], coor[0], time_info_str)
                            coor_list.append(coor_info)
                        else:
                            flag = False
                            break
                    if not flag:
                        continue
                    coor_list = np.array(coor_list)
                    coor_list_2 = coor_list[::2]
                    if len(coor_list_2) > 5:
                        final_length_list.append(len(coor_list_2))
                        f.write("#" + str(traj_count_id) + ":" + "\r")
                        f.write(">0:")
                        for j in coor_list_2:
                            f.write(j[0] + "," + j[1] + "," + j[2] + ";")
                        f.write("\r")
                        print("(" + str(whole_count) + ")" + "第" + str(traj_count_id) + "条原始轨迹写入完成")
                        traj_count_id += 1
                    else:
                        pass
                    if traj_count_id >= 20000:
                        break
                    whole_count += 1
    return final_length_list


if __name__ == '__main__':
    #boundary = (-8.665258, -8.528333, 41.10421, 41.24999) b4
    boundary=(-8.661858, -8.525016,41.06421, 41.20999)
    grid_size = 20
    out_path = './data/porto_2w_iter120_b5.txt'
    data_extract()
