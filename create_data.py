'''
create voxel data from lidar point cloud data
'''
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

def plot_point_cloud(pc, save_dir=None, name=None, show=True):
    '''
    pc: [N, 3]
    '''
    # 创建3D坐标轴
    ax = plt.axes(projection='3d')

    # 绘制点云
    sizes = np.ones((pc.shape[0])) * 2
    ax.scatter3D(pc[:, 0], pc[:, 1], pc[:, 2], s=sizes, cmap='Greens')

    # 添加标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    # 设置视角
    # 设置方位角为135度，俯仰角为30度
    ax.view_init(azim=135, elev=30)

    # 设置轴范围
    ax.set_xlim([0, 6.4])  # 设置 x 轴的范围
    ax.set_ylim([-3.2, 3.2])  # 设置 y 轴的范围
    ax.set_zlim([-2, 2])  # 设置 z 轴的范围
    # ax.set_xlim([2, 6])  # 设置 x 轴的范围
    # ax.set_ylim([-4, 2])  # 设置 y 轴的范围
    # ax.set_zlim([-2, 0.5])  # 设置 z 轴的范围

    # save the plot
    if save_dir is not None:
        if name is None:
            name = 'point_cloud'
        else:
            name = '{}_point_cloud'.format(name)
        plt.savefig(os.path.join(save_dir, name+'.svg'), format='svg')

    # 显示图形
    if show:
        plt.show()
    plt.close()


def plot_voxel(vl, save_dir=None, name=None, show=True):
    # 创建一个 Dx*Dy*Dz 的体素网格
    Dx, Dy, Dz = vl.shape
    voxels = np.zeros((Dx, Dy, Dz), dtype=bool)

    # 将有点云的体素进行可视化
    voxels[vl > 0] = True

    # 绘制体素数据
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(voxels, edgecolor='k')

    # 添加标签和标题
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel')

    # 设置坐标轴范围
    ax.set_xlim(0, Dx)
    ax.set_ylim(0, Dy)
    ax.set_zlim(0, Dz)

    # 设置视角
    # 设置方位角为135度，俯仰角为30度
    ax.view_init(azim=135, elev=30)

    # save the plot
    if save_dir is not None:
        if name is None:
            name = 'voxel'
        else:
            name = '{}_voxel'.format(name)
        plt.savefig(os.path.join(save_dir, name+'.png'))

    # 显示图形
    if show:
        plt.show()
    plt.close()


def ten2five(numb, length):
    # 将十进制数转换为五进制数
    digits = []
    while numb > 0:
        if numb > 5:
            numb, remainder = divmod(numb, 5)
            digits.append(remainder)
        else:
            digits.append(numb)
            break
    while len(digits) < length:
        digits.append(0)
    # 将五进制数表示为一个ndarray
    arr = np.array(digits[::-1])
    return arr

print(ten2five(125, 4))

def five2ten(arr):
    arr = arr[::-1]
    numb = np.sum(arr * (5 ** np.arange(len(arr))))
    return numb

print(five2ten(np.array([5, 5, 5])))

mmfi_root = '/data/szy4017/data/mmfi/E01/S01'
data_list = sorted(os.listdir(mmfi_root))
print(data_list)
all_file_list = []
for dl in data_list:
    file = os.path.join(mmfi_root, dl, 'lidar')
    file_list = sorted(os.listdir(file))
    for fl in file_list:
        all_file_list.append(os.path.join(mmfi_root, dl, 'lidar', fl))
print(all_file_list)
print(len(all_file_list))

# size_x, size_y, size_z = 32, 32, 20
size_x, size_y, size_z = 24, 64, 24
range_x = (2.0, 4.4)
range_y = (-3.2, 3.2)
range_z = (-1.4, 1.0)
# size_x, size_y, size_z = 64, 64, 40
# range_x = (0.0, 6.4)
# range_y = (-3.2, 3.2)
# range_z = (-2.0, 2.0)
for pc_file in all_file_list:
    print(pc_file)
    pc_arr = np.fromfile(pc_file, dtype=np.float64)
    pc_arr = pc_arr.reshape((-1, 3))
    print(pc_arr)
    print(pc_arr.shape)
    # plot_point_cloud(pc_arr)

    # convert point cloud to voxel
    scale_x = size_x / (range_x[1] - range_x[0])
    scale_y = size_y / (range_y[1] - range_y[0])
    scale_z = size_z / (range_z[1] - range_z[0])
    print(scale_x, scale_y, scale_z)

    pc_arr_translated = pc_arr - (range_x[0], range_y[0], range_z[0])
    pc_arr_scaled = pc_arr_translated * [scale_x, scale_y, scale_z]
    pc_arr_scaled = pc_arr_scaled.astype(int)
    print(pc_arr_scaled)
    unique_indices, counts = np.unique(pc_arr_scaled, return_counts=True, axis=0)
    print(unique_indices)
    print(counts)
    print(unique_indices.shape)
    print(counts.shape)

    voxel_grid = np.zeros((size_x, size_y, size_z))
    voxel_indices = unique_indices
    print(voxel_indices)
    voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = counts

    # for pa in pc_arr_scaled:
    #     voxel_grid[pa[0], pa[1], pa[2]] += 1
    # print(voxel_grid)
    print(voxel_grid.shape)
    x_flag = list(range(0, size_x+1, size_x//8))
    print(x_flag)
    voxel_map = []
    voxel_grid = np.clip(voxel_grid, None, 5)
    plot_voxel(voxel_grid)
    for i in range(8):
        voxel_part = voxel_grid[x_flag[i]:x_flag[i+1], :, :]
        print(voxel_part)
        print(voxel_part.shape)
        voxel_part_sum = voxel_part.sum(axis=0)
        indices = np.nonzero(voxel_part_sum)
        print(indices)
        map = np.zeros((size_y, size_z))
        for i in range(len(indices[0])):
            print(voxel_part[:, indices[0][i], indices[1][i]])
            numb = five2ten(voxel_part[:, indices[0][i], indices[1][i]])
            print(numb)
            map[indices[0][i], indices[1][i]] = numb
        map = map
        print(map.shape)
        plt.imshow(map)
        plt.show()
        voxel_map.append(map)
    voxel_map = np.stack(voxel_map, axis=2)
    # voxel_map = voxel_map.astype(np.uint8)
    # plt.imshow(voxel_map)
    # plt.show()

    voxel_recover = []
    for i in range(8):
        map = voxel_map[:, :, i]
        voxel_part_recover = np.zeros((size_x//8, size_y, size_z))
        indices = np.nonzero(map)
        print(indices)
        numb = map[indices[0], indices[1]]
        print(numb)
        for j, n in enumerate(numb):
            arr = ten2five(n, length=3)
            print(arr)
            voxel_part_recover[:, indices[0][j], indices[1][j]] = arr
        voxel_recover.append(voxel_part_recover)
    voxel_recover = np.vstack(voxel_recover)
    print(voxel_recover)
    print(voxel_recover.shape)
    print(voxel_grid == voxel_recover)
    plot_voxel(voxel_recover)
    pass


