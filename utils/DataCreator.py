# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : DataCreator.py
# @Time       : 2021/12/4 20:29

import random
import numpy as np


def sampling_func(input_matrix,
                  t_range,
                  x_range,
                  y_range=None,
                  boundary_num=50,
                  initial_num=50,
                  common_num=20000,
                  seed=None):

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
        total_boundary_num = len(input_matrix[0, :]) * 2
        total_common_num = np.prod(np.shape(input_matrix))

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
        initial_data_real = np.real(initial_data_list)
        initial_data_imag = np.imag(initial_data_list)

        initial_data = np.array(list(zip(initial_position_list,
                                         initial_time_list,
                                         initial_data_real,
                                         initial_data_imag)))

        # boundary data
        boundary_position_index_list = (boundary_data_index_list // len(time_list)) * -1
        boundary_time_index_list = boundary_data_index_list % len(time_list)

        boundary_position_list = x_position_list[boundary_position_index_list]
        boundary_time_list = time_list[boundary_time_index_list]
        boundary_data_list = input_matrix[boundary_position_index_list, boundary_time_index_list]
        boundary_data_real = np.real(boundary_data_list)
        boundary_data_imag = np.imag(boundary_data_list)

        boundary_data = np.array(list(zip(boundary_position_list,
                                          boundary_time_list,
                                          boundary_data_real,
                                          boundary_data_imag)))

        # common data
        common_data_position_index_list = common_data_index_list // len(time_list)
        common_data_time_index_list = common_data_index_list % len(time_list)

        common_data_position_list = x_position_list[common_data_position_index_list]
        common_data_time_list = time_list[common_data_time_index_list]
        common_data_list = input_matrix[common_data_position_index_list, common_data_time_index_list]
        common_data_real = np.real(common_data_list)
        common_data_imag = np.imag(common_data_list)

        common_data = np.array(list(zip(common_data_position_list,
                                        common_data_time_list,
                                        common_data_real,
                                        common_data_imag)))

        return {"initial": initial_data, "boundary": boundary_data, "common": common_data}
    elif len(input_matrix.shape) == 3 and y_range is not None:
        # x, y, t
        pass
    else:
        raise ValueError
