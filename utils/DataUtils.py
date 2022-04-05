# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : DataUtils.py
# @Time       : 2021/12/4 20:29

import random
import numpy as np


def basic_sampling_func(n_sampling,
                        sampling_range_conf,
                        sampling_type="random",
                        random_state=None):
    """
    基础采样函数
    :param n_sampling: 采样数
    :param sampling_range_conf: 采样范围以及采样步长
    :param sampling_type: 采样方法， 默认为完全随机采样
    :param random_state: 随机种子
    :return: 采样结果，按照传入的sampling_range_conf的顺序返回
    """

    # 生成采样区域
    linespace_list = [np.arange(sampling_conf[0], sampling_conf[1] + np.finfo(np.float32).eps, sampling_conf[2])
                      for sampling_conf in sampling_range_conf]

    sampling_space = np.meshgrid(*linespace_list)

    # 执行采样
    # 采样列表
    sampling_func_dict = {"random": random_sampling}
    sampling_result = sampling_func_dict[sampling_type](sampling_space, n_sampling, random_state)

    return sampling_result


def random_sampling(sampling_space, n_sampling, random_state):
    """
    随机采样函数
    :param sampling_space: 采样空间
    :param n_sampling: 采样数
    :param random_state: 随机种子
    :return: 采样结果
    """

    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # 转换为一维
    sampling_data_one_dim = [np.reshape(sampling_data, (-1,)) for sampling_data in sampling_space]
    # 随机采样
    sampling_data_num = len(sampling_data_one_dim[0])
    sampling_index = random.sample(range(sampling_data_num), n_sampling)

    sampling_result = [temp_sampling_data[sampling_index] for temp_sampling_data in sampling_data_one_dim]
    return sampling_result


def sampling_func(input_matrix,
                  t_range,
                  x_range,
                  y_range=None,
                  boundary_num=50,
                  initial_num=50,
                  common_num=20000,
                  periodic_boundary=False,
                  seed=None,
                  imag=True):
    """

    非周期边界条件：从上下边界随机采样，返回为（x, t, u）
    周期边界条件：仅对时间进行采样(upper_x, lower_x, t)

    :param input_matrix: 输入矩阵
    :param t_range: 时间范围
    :param x_range: x变量范围
    :param y_range: y变量范围（对于二维问题）
    :param boundary_num: 边界采样点数量
    :param initial_num: 初始条件采样点数量
    :param common_num: 一般点采样数量
    :param periodic_boundary: 是否为周期边界条件
    :param seed: 随机种子数
    :param imag: 是否返回虚数
    :return:
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    # data
    # boundary_data, initial_data, f_data
    # obtain from numerical method
    if len(input_matrix.shape) == 2:
        # x, t

        # cal total data num
        total_initial_num = len(input_matrix[:, 0])
        if periodic_boundary:
            total_boundary_num = len(input_matrix[0, :])
        else:
            total_boundary_num = len(input_matrix[0, :]) * 2
        total_common_num = np.prod(np.shape(input_matrix))

        # make sure the number < total sample points
        boundary_num = min(boundary_num, total_boundary_num)
        initial_num = min(initial_num, total_initial_num)
        common_num = min(common_num, int(total_common_num))

        # create true input list
        time_list = np.linspace(t_range[0], t_range[1], len(input_matrix[0, :]))
        x_position_list = np.linspace(x_range[0], x_range[1], len(input_matrix[:, 0]))

        # create choice data index
        initial_data_index_list = np.random.choice(range(0, total_initial_num), initial_num, replace=False)
        boundary_data_index_list = np.random.choice(range(0, total_boundary_num), boundary_num, replace=False)
        common_data_index_list = np.random.choice(range(0, total_common_num), common_num, replace=False)

        # create sampling data

        # initial data
        initial_position_list = x_position_list[initial_data_index_list]
        initial_time_list = np.ones_like(initial_position_list) * t_range[0]
        initial_data_list = input_matrix[initial_data_index_list, 0]

        if imag:
            initial_data_real = np.real(initial_data_list)
            initial_data_imag = np.imag(initial_data_list)

            initial_data = np.array(list(zip(initial_position_list,
                                             initial_time_list,
                                             initial_data_real,
                                             initial_data_imag)))
        else:
            initial_data_real = np.real(initial_data_list)

            initial_data = np.array(list(zip(initial_position_list,
                                             initial_time_list,
                                             initial_data_real)))

        # boundary data
        if periodic_boundary:
            boundary_time_index_list = boundary_data_index_list

            upper_boundary_position_list = np.ones(total_boundary_num) * x_position_list[-1]
            lower_boundary_position_list = np.ones(total_boundary_num) * x_position_list[0]

            boundary_time_list = time_list[boundary_time_index_list]

            boundary_data = np.array(list(zip(upper_boundary_position_list,
                                              lower_boundary_position_list,
                                              boundary_time_list)))
        else:
            boundary_position_index_list = (boundary_data_index_list // len(time_list)) * -1
            boundary_time_index_list = boundary_data_index_list % len(time_list)

            boundary_position_list = x_position_list[boundary_position_index_list]
            boundary_time_list = time_list[boundary_time_index_list]
            boundary_data_list = input_matrix[boundary_position_index_list, boundary_time_index_list]

            if imag:
                boundary_data_real = np.real(boundary_data_list)
                boundary_data_imag = np.imag(boundary_data_list)

                boundary_data = np.array(list(zip(boundary_position_list,
                                                  boundary_time_list,
                                                  boundary_data_real,
                                                  boundary_data_imag)))
            else:
                boundary_data_real = np.real(boundary_data_list)

                boundary_data = np.array(list(zip(boundary_position_list,
                                                  boundary_time_list,
                                                  boundary_data_real)))

        # common data
        common_data_position_index_list = common_data_index_list // len(time_list)
        common_data_time_index_list = common_data_index_list % len(time_list)

        common_data_position_list = x_position_list[common_data_position_index_list]
        common_data_time_list = time_list[common_data_time_index_list]
        common_data_list = input_matrix[common_data_position_index_list, common_data_time_index_list]

        if imag:
            common_data_real = np.real(common_data_list)
            common_data_imag = np.imag(common_data_list)

            common_data = np.array(list(zip(common_data_position_list,
                                            common_data_time_list,
                                            common_data_real,
                                            common_data_imag)))
        else:
            common_data_real = np.real(common_data_list)

            common_data = np.array(list(zip(common_data_position_list,
                                            common_data_time_list,
                                            common_data_real)))

        return {"initial": initial_data, "boundary": boundary_data, "common": common_data}
    elif len(input_matrix.shape) == 3 and y_range is not None:
        # x, y, t
        pass
    else:
        raise ValueError
