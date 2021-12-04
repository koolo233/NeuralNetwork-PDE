# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : shrodinger_equation_pinn.py
# @Time       : 2021/12/4 15:11
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from data.MyDataset import MyDataset
from utils.DataCreator import sampling_func
from models.PINN_models import PINN


class DataCreator(object):
    """
    create valuation data
    using finite difference with fourth Lunge-Kutta Method
    """

    def __init__(self, data_conf):
        super(DataCreator, self).__init__()

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

    def _data_init(self):

        # result matrix space_point * time_point
        self.phi_matrix = np.zeros((int(self.space_n), int(self.time_n))).astype(np.complex64)
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
        print("done...")

    def plot_func(self):
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
        ax.set_title("real part of wave function —— time")
        ax.set_xticklabels(time_labels)
        ax.set_yticklabels(position_labels)

        # imag
        plt.subplot(2, 1, 2)
        ax_imag = sns.heatmap(np.imag(phi_matrix_draw), annot=False)
        ax_imag.set_xlabel("time")
        ax_imag.set_ylabel("position")
        ax_imag.set_yticks(position_ticks)
        ax_imag.set_xticks(time_ticks)
        ax_imag.set_title("imaginary part of wave function —— time")
        ax_imag.set_xticklabels(time_labels)
        ax_imag.set_yticklabels(position_labels)

        plt.show()

    def sampling(self, boundary_num, initial_num, common_num, seed=None):
        x_range = [-self.box_l/2, self.box_l/2]
        t_range = [0, self.time_total]
        data_dict = sampling_func(self.phi_matrix, t_range, x_range, seed=seed,
                                  boundary_num=boundary_num,
                                  initial_num=initial_num,
                                  common_num=common_num)
        return data_dict


class ShrodingerEquationPinn(object):
    
    def __init__(self, conf):
        super(ShrodingerEquationPinn, self).__init__()

        # seed
        random.seed(conf["seed"])
        torch.manual_seed(conf["seed"])
        torch.random.manual_seed(conf["seed"])
        torch.cuda.manual_seed_all(conf["seed"])
        np.random.seed(conf["seed"])
        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # ---------------------------------------- #
        # ----------------- data ----------------- #
        # ---------------------------------------- #
        self.data_creator = DataCreator(conf)
        self.data_creator.iter_func()
        data_dict = self.data_creator.sampling(boundary_num=conf["boundary_num"],
                                               initial_num=conf["initial_num"],
                                               common_num=conf["common_num"], seed=conf["seed"])

        # raw data
        boundary_data = data_dict["boundary"]
        initial_data = data_dict["initial"]
        common_data = data_dict["common"]
        # dataset
        boundary_dataset = MyDataset(boundary_data)
        initial_dataset = MyDataset(initial_data)
        common_dataset = MyDataset(common_data)
        # dataloader
        if conf["boundary_batch_size"] == -1:
            self.boundary_dataloader = DataLoader(boundary_dataset, batch_size=len(boundary_dataset))
        else:
            self.boundary_dataloader = DataLoader(boundary_dataset, batch_size=conf["boundary_batch_size"])

        if conf["initial_batch_size"] == -1:
            self.initial_dataloader = DataLoader(initial_dataset, batch_size=len(initial_dataset))
        else:
            self.initial_dataloader = DataLoader(initial_dataset, batch_size=conf["initial_batch_size"])

        if conf["common_batch_size"] == -1:
            self.common_dataloader = DataLoader(common_dataset, batch_size=len(common_dataset))
        else:
            self.common_dataloader = DataLoader(common_dataset, batch_size=conf["common_batch_size"])

        # ----------------------------------------- #
        # ----------------- model ----------------- #
        # ----------------------------------------- #
        self.pinn_model = PINN(input_dim=2, output_dim=2, dim_list=conf["model_layers"])

        # ------------------------------------------- #
        # ----------------- recoder ----------------- #
        # ------------------------------------------- #


if __name__ == "__main__":
    _conf = yaml.load(open("./conf/pinn_shrodinger_equation.yaml"), Loader=yaml.FullLoader)
    data_creator = DataCreator(_conf)
    data_creator.iter_func()
    data_creator.plot_func()
