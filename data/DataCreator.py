# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : DataUtils.py
# @Time       : 2021/12/6 13:18


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.DataUtils import sampling_func, basic_sampling_func
from utils.PlotUtils import heatmap_plot_func, two_dim_curve_gif_func


class BasicDataCreator(object):

    def _data_init(self):
        pass

    def iter_func(self):
        pass

    def plot_func(self, *args):
        pass

    def sampling(self, *args):
        pass


class OneDAllenCahnEquationDataCreator(BasicDataCreator):

    def __init__(self, data_conf):
        super(OneDAllenCahnEquationDataCreator, self).__init__()

        self.conf = data_conf
        # cal area length
        self.cal_x_range = data_conf["cal_x_range"]
        # cal time
        self.time_total = data_conf["time_total"]
        # time step
        self.delta_time = data_conf["delta_time"]
        # space step
        self.delta_x = data_conf["delta_x"]

        # time discrete num
        self.time_n = int(self.time_total / self.delta_time)
        # space discrete num
        self.space_n = int((self.cal_x_range[1] - self.cal_x_range[0]) / self.delta_x)

        # result matrix
        self.result_matrix = None

        # data init
        self._data_init()

        # output
        self.figure_output_path = os.path.join(data_conf["figure_output_root"],
                                               data_conf["numerical_figure_output_name"])
        self.gif_output_path = os.path.join(data_conf["figure_output_root"],
                                            data_conf["numerical_gif_output_name"])

    def _data_init(self):
        self.result_matrix = np.zeros((self.space_n, self.time_n)).astype(np.float32)

        # init
        x_list = np.linspace(self.cal_x_range[0], self.cal_x_range[1], self.space_n)
        self.result_matrix[:, 0] = x_list ** 2 * np.cos(np.pi * x_list)

    def iter_func(self):
        pass

    def plot_func(self, plot_figure=False, save_figure=False, cmap="rainbow"):
        x_value_list = np.linspace(0, self.time_total, self.time_n)
        y_value_list = np.linspace(self.cal_x_range[0], self.cal_x_range[1], self.space_n)
        heatmap_plot_func(data_matrix=self.result_matrix,
                          draw_size=[400, 400],
                          x_value_list=x_value_list,
                          y_value_list=y_value_list,
                          figsize=(15, 5),
                          figure_output_path=self.figure_output_path,
                          title=self.conf["func_name"],
                          xlabel="time",
                          ylabel="position",
                          cmap=cmap
                          )
        two_dim_curve_gif_func(data_matrix=self.result_matrix,
                               x_value_list=x_value_list,
                               y_value_list=y_value_list,
                               figure_size=(10, 10),
                               split_alone_axis=1,
                               gif_frame_num=150,
                               title=self.conf["func_name"],
                               xlabel="x",
                               ylabel="u(x, t)",
                               gif_output_path=self.gif_output_path,
                               figure_dpi=100,
                               x_limits="fixed",
                               y_limits="fixed"
                               )

    def sampling(self, boundary_num, initial_num, common_num, seed=None, periodic_boundary=False):
        t_range = [0, self.time_total]
        data_dict = sampling_func(self.result_matrix, t_range, self.cal_x_range, seed=seed,
                                  boundary_num=boundary_num,
                                  initial_num=initial_num,
                                  common_num=common_num,
                                  imag=False,
                                  periodic_boundary=periodic_boundary)
        return data_dict


class OneDHeatTransferEquationDataCreator(BasicDataCreator):

    def __init__(self, data_conf):
        super(OneDHeatTransferEquationDataCreator, self).__init__()

        self.conf = data_conf
        # cal area length
        self.cal_x_range = data_conf["cal_x_range"]
        # cal time
        self.time_total = data_conf["time_total"]
        # time step
        self.delta_time = data_conf["delta_time"]
        # space step
        self.delta_x = data_conf["delta_x"]

        # time discrete num
        self.time_n = int(self.time_total / self.delta_time)
        # space discrete num
        self.space_n = int((self.cal_x_range[1] - self.cal_x_range[0]) / self.delta_x)

        # result matrix
        self.result_matrix = None
        # param matrix
        self.parm_matrix = None
        # k matrix
        self.k_vector_matrix = None

        # data init
        self._data_init()

        # output

        if not os.path.exists(data_conf["figure_output_root"]):
            os.makedirs(data_conf["figure_output_root"])
        self.figure_output_path = os.path.join(data_conf["figure_output_root"],
                                               data_conf["numerical_figure_output_name"])
        self.gif_output_path = os.path.join(data_conf["figure_output_root"],
                                            data_conf["numerical_gif_output_name"])

    def _data_init(self):
        self.result_matrix = np.zeros((self.space_n, self.time_n)).astype(np.float32)

        # init
        self.result_matrix[:, 0] = np.sin(np.pi * np.linspace(self.cal_x_range[0], self.cal_x_range[1], self.space_n)) * 200
        # self.result_matrix[:, 0] = 200
        # self.result_matrix[-1, 0] = 0
        # self.result_matrix[0, 0] = 0

        # def A matrix
        self.parm_matrix = -2 * np.eye(int(self.space_n)) + \
                           np.eye(int(self.space_n), k=1) + \
                           np.eye(int(self.space_n), k=-1)
        self.parm_matrix[0, :] = 0
        self.parm_matrix[-1, :] = 0

        # def k1, k2, k3, k4
        # k1 = k_vector_matrix[:, 0]
        # k2 = k_vector_matrix[:, 1]
        # k3 = k_vector_matrix[:, 2]
        # k4 = k_vector_matrix[:, 3]
        self.k_vector_matrix = np.zeros((int(self.space_n), 4))

    def iter_func(self):
        constant_ = 0.1 / np.power(self.delta_x, 2)
        for i in range(self.time_n - 1):
            # k1
            self.k_vector_matrix[:, 0] = constant_ * np.matmul(self.parm_matrix, self.result_matrix[:, i])
            # k2
            self.k_vector_matrix[:, 1] = constant_ * (
                np.matmul(self.parm_matrix, self.result_matrix[:, i] +
                          (self.delta_time / 2) * self.k_vector_matrix[:, 0]))
            # k3
            self.k_vector_matrix[:, 2] = constant_ * (
                np.matmul(self.parm_matrix, self.result_matrix[:, i] +
                          (self.delta_time / 2) * self.k_vector_matrix[:, 1]))
            # k4
            self.k_vector_matrix[:, 3] = constant_ * (
                np.matmul(self.parm_matrix, self.result_matrix[:, i] +
                          self.delta_time * self.k_vector_matrix[:, 2]))
            # if i % 1000 == 0:
            #     print(np.max(k_vector_matrix))
            self.result_matrix[:, i + 1] = self.result_matrix[:, i] + (self.delta_time / 6) * (
                    self.k_vector_matrix[:, 0] + 2 * self.k_vector_matrix[:, 1] +
                    2 * self.k_vector_matrix[:, 2] + self.k_vector_matrix[:, 3])

    def plot_func(self, plot_figure=False, save_figure=False):
        x_value_list = np.linspace(0, self.time_total, self.time_n)
        y_value_list = np.linspace(self.cal_x_range[0], self.cal_x_range[1], self.space_n)
        heatmap_plot_func(data_matrix=self.result_matrix,
                          draw_size=[400, 400],
                          x_value_list=x_value_list,
                          y_value_list=y_value_list,
                          figsize=(15, 5),
                          figure_output_path=self.figure_output_path,
                          title=self.conf["func_name"],
                          xlabel="time",
                          ylabel="position"
                          )
        two_dim_curve_gif_func(data_matrix=self.result_matrix,
                               x_value_list=x_value_list,
                               y_value_list=y_value_list,
                               figure_size=(10, 10),
                               split_alone_axis=1,
                               gif_frame_num=150,
                               title=self.conf["func_name"],
                               xlabel="x",
                               ylabel="u(x, t)",
                               gif_output_path=self.gif_output_path,
                               figure_dpi=100,
                               x_limits="fixed",
                               y_limits="fixed"
                               )

    def sampling(self, boundary_num, initial_num, common_num, seed=None):
        t_range = [0, self.time_total]
        data_dict = sampling_func(self.result_matrix, t_range, self.cal_x_range, seed=seed,
                                  boundary_num=boundary_num,
                                  initial_num=initial_num,
                                  common_num=common_num, imag=False)
        return data_dict


class BurgersEquationDataCreator(BasicDataCreator):

    def __init__(self, data_conf):
        super(BurgersEquationDataCreator, self).__init__()

        # cal area length
        self.cal_x_range = data_conf["cal_x_range"]
        # cal time
        self.time_total = data_conf["time_total"]
        # time step
        self.delta_time = data_conf["delta_time"]
        # space step
        self.delta_x = data_conf["delta_x"]

        # time discrete num
        self.time_n = int(self.time_total / self.delta_time)
        # space discrete num
        self.space_n = int((self.cal_x_range[1] - self.cal_x_range[0]) / self.delta_x)

        # result matrix
        self.result_matrix = None

        # data init
        self._data_init()

        # output
        self.figure_output_path = os.path.join(data_conf["figure_output_root"],
                                               data_conf["numerical_figure_output_name"])
        self.gif_output_path = os.path.join(data_conf["figure_output_root"],
                                            data_conf["numerical_gif_output_name"])

    def _data_init(self):
        self.result_matrix = np.zeros((self.space_n, self.time_n))

        # init wave
        x_list = np.linspace(self.cal_x_range[0], self.cal_x_range[1], self.space_n)
        self.result_matrix[:, 0] = -np.sin(np.pi * x_list)

    def iter_func(self):
        pass

    def plot_func(self, plot_figure=False, save_figure=False):
        x_value_list = np.linspace(0, self.time_total, self.time_n)
        y_value_list = np.linspace(self.cal_x_range[0], self.cal_x_range[1], self.space_n)
        heatmap_plot_func(data_matrix=self.result_matrix,
                          draw_size=[400, 400],
                          x_value_list=x_value_list,
                          y_value_list=y_value_list,
                          figsize=(15, 5),
                          figure_output_path=self.figure_output_path,
                          title="Burgers_equation",
                          xlabel="time",
                          ylabel="position",
                          x_ticks_num=3,
                          )
        two_dim_curve_gif_func(data_matrix=self.result_matrix,
                               x_value_list=x_value_list,
                               y_value_list=y_value_list,
                               figure_size=(10, 10),
                               split_alone_axis=1,
                               gif_frame_num=150,
                               title="Burgers_equation",
                               xlabel="x",
                               ylabel="u(x, t)",
                               gif_output_path=self.gif_output_path,
                               figure_dpi=100,
                               x_limits="fixed",
                               y_limits="fixed"
                               )

    def sampling(self, boundary_num, initial_num, common_num, seed=None):
        t_range = [0, self.time_total]
        data_dict = sampling_func(self.result_matrix, t_range, self.cal_x_range, seed=seed,
                                  boundary_num=boundary_num,
                                  initial_num=initial_num,
                                  common_num=common_num, imag=False)
        return data_dict


class ShrodingerEquationDataCreator(BasicDataCreator):
    """
    create valuation data
    using finite difference with fourth Lunge-Kutta Method
    """

    def __init__(self, data_conf):
        super(ShrodingerEquationDataCreator, self).__init__()

        # state num
        self.state_num = data_conf['state_num']
        # box length
        self.box_l = data_conf['box_l']
        # cal time
        self.time_total = data_conf['time_total']
        # time step
        self.delta_time = data_conf['delta_time']
        # space step
        self.delta_x = data_conf['delta_x']
        # time discrete num
        self.time_n = int(self.time_total / self.delta_time)
        # space discrete num
        self.space_n = int(self.box_l / self.delta_x)

        # result matrix
        self.phi_matrix = None
        # param matrix
        self.parm_matrix = None
        # k matrix
        self.k_vector_matrix = None

        # data init
        self._data_init()

        # output
        self.figure_output_path = os.path.join(data_conf["figure_output_root"],
                                               data_conf["numerical_figure_output_name"])

    def _data_init(self):

        # result matrix space_point * time_point
        self.phi_matrix = np.zeros((self.space_n, self.time_n)).astype(np.complex64)
        # def A matrix
        self.parm_matrix = -2 * np.eye(int(self.space_n)) + \
                           np.eye(int(self.space_n), k=1) + \
                           np.eye(int(self.space_n), k=-1) + 0.j
        self.parm_matrix[0, :] = 0
        self.parm_matrix[-1, :] = 0

        # init wave
        self.phi_matrix[:, 0] = np.sin((self.state_num * np.pi / self.box_l) *
                                       (np.linspace(-self.box_l / 2, self.box_l / 2, self.space_n) +
                                        self.box_l / 2))

        # def k1, k2, k3, k4
        # k1 = k_vector_matrix[:, 0]
        # k2 = k_vector_matrix[:, 1]
        # k3 = k_vector_matrix[:, 2]
        # k4 = k_vector_matrix[:, 3]
        self.k_vector_matrix = np.zeros((int(self.space_n), 4)).astype(np.complex64)

    def iter_func(self):
        constant_ = 1.j / (2 * np.power(self.delta_x, 2))
        for i in range(self.time_n - 1):
            # k1
            self.k_vector_matrix[:, 0] = constant_ * np.matmul(self.parm_matrix, self.phi_matrix[:, i])
            # k2
            self.k_vector_matrix[:, 1] = constant_ * (
                np.matmul(self.parm_matrix, self.phi_matrix[:, i] +
                          (self.delta_time / 2) * self.k_vector_matrix[:, 0]))
            # k3
            self.k_vector_matrix[:, 2] = constant_ * (
                np.matmul(self.parm_matrix, self.phi_matrix[:, i] +
                          (self.delta_time / 2) * self.k_vector_matrix[:, 1]))
            # k4
            self.k_vector_matrix[:, 3] = constant_ * (
                np.matmul(self.parm_matrix, self.phi_matrix[:, i] +
                          self.delta_time * self.k_vector_matrix[:, 2]))
            # if i % 1000 == 0:
            #     print(np.max(k_vector_matrix))
            self.phi_matrix[:, i + 1] = self.phi_matrix[:, i] + (self.delta_time / 6) * (
                        self.k_vector_matrix[:, 0] + 2 * self.k_vector_matrix[:, 1] +
                        2 * self.k_vector_matrix[:, 2] + self.k_vector_matrix[:, 3])

    def plot_func(self, plot_figure=False, save_figure=False):
        # plot

        plt.figure(figsize=(10, 10), dpi=150)

        draw_time_list = np.linspace(0, self.time_n - 1, min(400, self.time_n)).astype(np.int32)
        draw_position_list = np.linspace(0, self.space_n - 1, min(400, self.space_n)).astype(np.int32)
        phi_matrix_draw = self.phi_matrix[draw_position_list, :][:, draw_time_list]

        time_list = np.linspace(0, self.time_total, len(draw_time_list))
        position_list = np.linspace(-self.box_l / 2, self.box_l / 2, len(draw_position_list))

        position_labels = np.around(np.linspace(-self.box_l / 2, self.box_l / 2, 4), 1)
        # the index position of the tick labels
        position_ticks = list()
        for label in position_labels:
            idx_pos = len(position_list) - np.argmin(np.abs(label - position_list))
            position_ticks.append(idx_pos)

        time_labels = np.around(np.linspace(0, self.time_total, 4), 1)
        time_ticks = list()
        for label in time_labels:
            idx_pos = np.argmin(np.abs(label - time_list))
            time_ticks.append(idx_pos)

        # real
        plt.subplot(2, 1, 1)
        ax = sns.heatmap(np.real(phi_matrix_draw), annot=False)
        ax.set_xlabel("time")
        ax.set_ylabel("position")
        ax.set_yticks(position_ticks)
        ax.set_xticks(time_ticks)
        ax.set_title("real part of wave function ?????? time")
        ax.set_xticklabels(time_labels)
        ax.set_yticklabels(position_labels)

        # imag
        plt.subplot(2, 1, 2)
        ax_imag = sns.heatmap(np.imag(phi_matrix_draw), annot=False)
        ax_imag.set_xlabel("time")
        ax_imag.set_ylabel("position")
        ax_imag.set_yticks(position_ticks)
        ax_imag.set_xticks(time_ticks)
        ax_imag.set_title("imaginary part of wave function ?????? time")
        ax_imag.set_xticklabels(time_labels)
        ax_imag.set_yticklabels(position_labels)

        if save_figure:
            plt.savefig(self.figure_output_path)
        if plot_figure:
            plt.show()

    def sampling(self, boundary_num, initial_num, common_num, seed=None):
        x_range = [-self.box_l/2, self.box_l/2]
        t_range = [0, self.time_total]
        data_dict = sampling_func(self.phi_matrix, t_range, x_range, seed=seed,
                                  boundary_num=boundary_num,
                                  initial_num=initial_num,
                                  common_num=common_num)
        return data_dict


class OneDHeatMultiLayerDataCreator(BasicDataCreator):

    def __init__(self, data_conf):
        super(OneDHeatMultiLayerDataCreator, self).__init__()

        self.conf = data_conf
        # compute area length
        self.compute_length_list = data_conf["length_list"]
        assert len(self.compute_length_list) == 2
        # space total length
        self.space_length = np.sum(self.compute_length_list)
        # compute time
        self.time_total = data_conf["time_total"]
        # time step
        self.delta_time = data_conf["delta_time"]
        # space step
        self.delta_space = data_conf["delta_space"]

        # time discrete num
        self.time_n = int(self.time_total / self.delta_time)
        # space discrete num
        self.space_n = int(np.sum(self.compute_length_list) / self.delta_space)

        # result matrix
        self.result_matrix = None
        # param matrix
        self.parm_matrix = None
        # k matrix
        self.k_vector_matrix = None

        # data init
        self._data_init()

        # # output
        # if not os.path.exists(data_conf["figure_output_root"]):
        #     os.makedirs(data_conf["figure_output_root"])
        # self.figure_output_path = os.path.join(data_conf["figure_output_root"],
        #                                        data_conf["numerical_figure_output_name"])
        # self.gif_output_path = os.path.join(data_conf["figure_output_root"],
        #                                     data_conf["numerical_gif_output_name"])

    def sampling(self):

        # ????????????????????????

        # ??????????????????????????????????????????????????????
        # ????????????conf
        time_conf = [0, 0, self.delta_time]
        space_conf = [0, self.space_length, self.delta_space]
        init_cond_sampling_conf = [space_conf, time_conf]

        init_cond_sampling_data = basic_sampling_func(self.conf["initial_num"],
                                                      init_cond_sampling_conf)

        # ??????????????????????????????????????????????????????
        # ????????????conf
        time_conf = [0, self.time_total, self.delta_time]
        # ????????????
        space_conf_1 = [0, 0, self.delta_space]
        boundary_cond_sampling_conf_1 = [space_conf_1, time_conf]

        space_conf_2 = [self.space_length, self.space_length, self.delta_space]
        boundary_cond_sampling_conf_2 = [space_conf_2, time_conf]

        boundary_cond_sampling_data_1 = basic_sampling_func(int(self.conf["boundary_num"]/2),
                                                            boundary_cond_sampling_conf_1)
        boundary_cond_sampling_data_2 = basic_sampling_func(int(self.conf["boundary_num"] / 2),
                                                            boundary_cond_sampling_conf_2)
        boundary_cond_sampling_data = [np.hstack((boundary_cond_sampling_data_1[i],
                                                 boundary_cond_sampling_data_2[i])) for i in range(2)]

        # ??????????????????????????????????????????PDE
        time_conf = [0, self.time_total, self.delta_time]
        space_conf = [0, self.space_length, self.delta_space]
        pde_cond_sampling_conf = [space_conf, time_conf]
        pde_cond_sampling_data = basic_sampling_func(self.conf["common_num"],
                                                     pde_cond_sampling_conf)

        # ??????????????????????????????????????????????????????
        time_conf = [0, self.time_total, self.delta_time]
        space_conf = [self.compute_length_list[0], self.compute_length_list[0], self.delta_space]
        interface_cond_sampling_conf = [space_conf, time_conf]
        interface_sampling_data = basic_sampling_func(self.conf["interface_num"],
                                                      interface_cond_sampling_conf)

        sampling_dict = dict()
        sampling_dict["initial_data"] = init_cond_sampling_data
        sampling_dict["pde_data"] = pde_cond_sampling_data
        sampling_dict["boundary_data"] = boundary_cond_sampling_data
        sampling_dict["interface_data"] = interface_sampling_data
        return sampling_dict


if __name__ == "__main__":
    test_config = {"length_list": [1, 1],
                   "time_total": 2,
                   "delta_time": 0.01,
                   "delta_space": 0.01,
                   "initial_num": 10,
                   "boundary_num": 30,
                   "common_num": 20,
                   "interface_num": 20}
    test_data_creator = OneDHeatMultiLayerDataCreator(test_config)

    samples = test_data_creator.sampling()
    print(samples)

