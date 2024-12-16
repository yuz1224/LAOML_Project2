from matplotlib import pyplot as plt
from fnn import visualization_2D
from deeponet import DeepONet
import numpy as np


# Poisson 2D data generation
def generate_deeponet_data(a_list, num_sample_points=50, num_query_points=2000):
    """
    Generate data for DeepONet where input function sampling points and query points are different.

    我们的偏微分方程 里面有两个参数不确定, fnn只能针对单个a值或者b值
    我们训练的deepONET这个架构是可以针对多个a值或者b值
    #TODO 目前我们这里只考虑了a值的变化, 在a_list里面

    1.  DeepONET的branch网络将函数f作为参数, f与a有关, 我们会考虑a_list中的所有a值, 得到多个f
        每一个 f 需要 进行采样, 采出其中的离散点, 因此 Branch_input的size为 {num_sample_points^2, len(a_list)}.

    2. DeepONET的trunk网络, 接受我们想要预测的 二维函数 u 的 自变量(x,y) 作为参数,
        我们一般会要多个点, 因此 Trunk_input的size 为 {2,num_query_points^2}.

    3. DeepONET的输出 为 u(x,y)的值, 因为 u也与a的取值有关, 然后我们需要预测出 trunk网络输入的所有坐标对应的 u(x,y)
        因此输出 {labels} 的 size 为 {len(a_list), num_query_points^2}.

    """

    # 生成 Branch 网络输入: 输入函数的离散采样点
    input_x = np.linspace(0, 1, num_sample_points)
    input_y = np.linspace(0, 1, num_sample_points)
    input_X, input_Y = np.meshgrid(input_x, input_y)
    branch_points = np.c_[input_X.ravel(), input_Y.ravel()]

    # 生成 Trunk 网络输入: 独立的查询点网格
    trunk_input = np.random.uniform(0, 1, [num_query_points, 2])

    # 生成 batch_size 个不同参数 a 的函数 f 和目标 u
    branch_input = []
    labels = []

    for a in a_list:
        # f(x, y) = 2 * a * π^2 * sin(πx) * sin(πy)，在 f(x,y) 上采样
        f = 2 * a * (np.pi ** 2) * np.sin(np.pi * branch_points[:, 0]) * np.sin(np.pi * branch_points[:, 1])
        branch_input.append(f)

        # u(x, y) = a * sin(πx) * sin(πy)，在查询点 trunk_input 上采样
        u = a * np.sin(np.pi * trunk_input[:, 0]) * np.sin(np.pi * trunk_input[:, 1])
        labels.append(u)

    # 转换为 NumPy 数组
    branch_input = np.stack(branch_input, axis=0)  # Shape: (batch_size, num_input_points)
    labels = np.stack(labels, axis=0)             # Shape: (batch_size, num_query_points)

    return branch_input.T, branch_points.T, trunk_input.T, labels

# Data preparation
a_train = np.linspace(-1, 1, 1000) # Training values for a
a_test = np.array([np.random.uniform(-1, 1) for _ in range(10)])
# a_train = [1]
# a_test = [1]

## Generating data
x_branch_train, branch_points_train, x_trunk_train, u_train = generate_deeponet_data(a_train)
x_branch_test, branch_points_test, x_trunk_test, u_test  = generate_deeponet_data(a_test)

# In[] 可视化生成的数据
# visualize f(x,y)
a_list_index = -1
f_values = x_branch_train[:, a_list_index]  # 取第0个样本
branch_x, branch_y = branch_points_train[0, :], branch_points_train[1, :]
visualization_2D(branch_x, branch_y, f_values, title=f"Input Function f(x, y)")

# visualize u(x,y), 根据这个来判断应该没错
u_values_sample = u_train[a_list_index, :]  # Shape: (num_query_points,)
trunk_x, trunk_y = x_trunk_train[0, :], x_trunk_train[1, :]  # Coordinates
visualization_2D(trunk_x, trunk_y, u_values_sample, title=f"Output Function u(x, y)")

# In[]
# Shuffle training data
sample_num, func_num = x_branch_train.shape
permute_idx = np.random.permutation(func_num)
x_branch_train = x_branch_train[:, permute_idx]
u_train = u_train[permute_idx]

# In[] DeepONet setup

branch_layer_sizes = [sample_num, 32, 64]  # Branch net: takes a_i as input
trunk_layer_sizes = [2, 32, 64]  # Trunk net: takes (x, y) as input
deeponet = DeepONet(branch_layer_sizes, trunk_layer_sizes, "relu")

# Train DeepONet
# x_branch_train, branch_points_train, x_trunk_train, u_train
# x_branch_test, branch_points_test, x_trunk_test, u_test

record = deeponet.train(x_branch_train, x_trunk_train, u_train, epochs=1000,
                        learning_rate=0.0001, batch_size=64, opt_type="adam",
                        x_val=(x_branch_test, x_trunk_test), u_val=u_test)

u_test_pred = deeponet.infer(x_branch_test, x_trunk_test)
trunk_x, trunk_y = x_trunk_test[0, :], x_trunk_test[1, :]  # Coordinates
visualization_2D(trunk_x, trunk_y, u_test_pred[0], title=f"Predicted Output u(x, y)")
visualization_2D(trunk_x, trunk_y, u_test[0], title=f"Actual Output u(x, y)")
visualization_2D(trunk_x, trunk_y, np.abs(u_test_pred[0] - u_test[0]), title=f"Error in u(x, y)")