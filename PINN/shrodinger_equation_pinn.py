# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : shrodinger_equation_pinn.py
# @Time       : 2021/12/4 15:11
import os
import sys
sys.path.append(os.curdir)  # add ROOT to PATH
import time
import yaml
import random
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data.MyDataset import MyDataset
from utils.DataCreator import sampling_func
from models.PINN_models import PINN


logging.basicConfig(level=logging.NOTSET,
                    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create file handler
run_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_file_path = os.path.join("./logs", "{}.log".format(run_time))
fh = logging.FileHandler(log_file_path, mode="w")
fh.setLevel(logging.NOTSET)

# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# output format
basic_format = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(basic_format)
# ch.setFormatter(basic_format)

# add logger to handler
logger.addHandler(fh)
# logger.addHandler(ch)


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

        # output
        self.figure_output_path = os.path.join(data_conf["figure_output_root"],
                                               data_conf["numerical_figure_output_name"])

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


class ShrodingerEquationPinn(object):
    
    def __init__(self, conf):
        super(ShrodingerEquationPinn, self).__init__()
        logger.info("PINN for Shrodinger Equation \n \n \n")
        logger.info("hyps list {}".format(conf))

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
        self.EPOCHS = conf["max_epochs"]

        # ---------------------------------------- #
        # ----------------- data ----------------- #
        # ---------------------------------------- #
        self.data_creator = DataCreator(conf)
        begin_fd_time = time.time()
        logger.info("create data ...")
        self.data_creator.iter_func()
        over_fd_time = time.time()
        logger.info("finite difference with fourth lunge-kutta method uses {:.5f}s".format(over_fd_time - begin_fd_time))
        self.data_creator.plot_func(save_figure=True)
        logger.info("save figure as {}".format(self.data_creator.figure_output_path))

        # sample
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
            self.boundary_dataloader = DataLoader(boundary_dataset, batch_size=conf["boundary_num"],
                                                  collate_fn=boundary_dataset.collate_fn)
        else:
            self.boundary_dataloader = DataLoader(boundary_dataset, batch_size=conf["boundary_batch_size"],
                                                  collate_fn=boundary_dataset.collate_fn)

        if conf["initial_batch_size"] == -1:
            self.initial_dataloader = DataLoader(initial_dataset, batch_size=conf["initial_num"],
                                                 collate_fn=initial_dataset.collate_fn)
        else:
            self.initial_dataloader = DataLoader(initial_dataset, batch_size=conf["initial_batch_size"],
                                                 collate_fn=initial_dataset.collate_fn)

        if conf["common_batch_size"] == -1:
            self.common_dataloader = DataLoader(common_dataset, batch_size=conf["common_num"],
                                                collate_fn=common_dataset.collate_fn)
        else:
            self.common_dataloader = DataLoader(common_dataset, batch_size=conf["common_batch_size"],
                                                collate_fn=common_dataset.collate_fn)

        logger.info("create dataset and dataloader done ...")

        # ----------------------------------------- #
        # ----------------- model ----------------- #
        # ----------------------------------------- #
        self.pinn_model = PINN(input_dim=2, output_dim=2, dim_list=conf["model_layers"]).to(self.device)
        logger.info("create pinn done ...")

        # ------------------------------------------------------ #
        # ----------------- optimizer and loss ----------------- #
        # ------------------------------------------------------ #
        params = [p for p in self.pinn_model.parameters() if p.requires_grad]
        # self.optimizer = optim.LBFGS(params, lr=conf["lr"],
        #                              max_iter=conf["max_iter"],
        #                              max_eval=conf["max_eval"],
        #                              history_size=conf["history_size"],
        #                              tolerance_grad=conf["tolerance_grad"])
        self.optimizer = optim.Adam(params)
        # self.optimizer = optim.SGD(params, lr=5e-4)
        logger.info("create optimizer and criterion done ...")

        # ---------------------------------------- #
        # ----------------- save ----------------- #
        # ---------------------------------------- #
        self.model_output_path = os.path.join(conf["model_output_root"], conf["model_output_name"])
        self.pinn_val_output_figure = os.path.join(conf["figure_output_root"], conf["pinn_figure_output_name"])

    def train(self):

        logger.info("begin train ...")
        begin_train_time = time.time()

        for epoch in range(1, self.EPOCHS+1):
            self.pinn_model.train()
            self.optimizer.zero_grad()

            boundary_loss = np.zeros(1)
            initial_loss = np.zeros(1)
            common_loss = np.zeros(1)

            # boundary data
            for step, batch in enumerate(self.boundary_dataloader):
                boundary_input_tensor_list = batch["input"]
                boundary_input_x = boundary_input_tensor_list[0].to(self.device)
                boundary_input_t = boundary_input_tensor_list[1].to(self.device)

                boundary_pred_tensor = self.pinn_model(boundary_input_x, boundary_input_t)

                boundary_loss = torch.mean((boundary_pred_tensor -
                                            torch.zeros_like(boundary_pred_tensor).to(self.device))**2)

                boundary_loss.backward()

            # initial data
            for step, batch in enumerate(self.initial_dataloader):
                initial_input_tensor_list = batch["input"]
                initial_output_tensor = batch["output"].to(self.device)
                initial_input_x = initial_input_tensor_list[0].to(self.device)
                initial_input_t = initial_input_tensor_list[1].to(self.device)

                initial_pred_tensor = self.pinn_model(initial_input_x, initial_input_t)

                initial_loss = torch.mean((initial_pred_tensor-initial_output_tensor)**2)

                initial_loss.backward()

            # common data
            for step, batch in enumerate(self.common_dataloader):
                common_input_tensor_list = batch["input"]
                common_input_x = common_input_tensor_list[0].to(self.device)
                common_input_t = common_input_tensor_list[1].to(self.device)

                mask_matrix = torch.ones(common_input_x.shape[0]).to(self.device)

                # y
                common_pred_tensor = self.pinn_model(common_input_x, common_input_t)

                # dy/dx
                dy_dx_real = torch.autograd.grad(common_pred_tensor[:, 0], common_input_x,
                                                 grad_outputs=mask_matrix,
                                                 create_graph=True, retain_graph=True)[0]
                dy_dx_imag = torch.autograd.grad(common_pred_tensor[:, 1], common_input_x,
                                                 grad_outputs=mask_matrix,
                                                 create_graph=True, retain_graph=True)[0]

                # dy/dt
                dy_dt_real = torch.autograd.grad(common_pred_tensor[:, 0], common_input_t,
                                                 grad_outputs=mask_matrix,
                                                 create_graph=True, retain_graph=True)[0]
                dy_dt_imag = torch.autograd.grad(common_pred_tensor[:, 1], common_input_t,
                                                 grad_outputs=mask_matrix,
                                                 create_graph=True, retain_graph=True)[0]

                # d^2y/dx^2
                dy_dx_real_2nd = torch.autograd.grad(dy_dx_real, common_input_x,
                                                     grad_outputs=mask_matrix,
                                                     create_graph=True, retain_graph=True)[0]
                dy_dx_imag_2nd = torch.autograd.grad(dy_dx_imag, common_input_x,
                                                     grad_outputs=mask_matrix,
                                                     create_graph=True, retain_graph=True, allow_unused=True)[0]
                # PDE output
                pde_output_real = 2*dy_dx_real_2nd - dy_dt_imag
                pde_output_imag = 2*dy_dx_imag_2nd + dy_dt_real

                common_loss = torch.mean(pde_output_real**2)+torch.mean(pde_output_imag**2)

                common_loss.backward()

            self.optimizer.step()

            if epoch % 10 == 0:
                logger.info("[{}/{}]\tboundary loss:{:.5f}\tinitial loss:{:.5f}\tcommon loss:{:.5f}".format(epoch,
                                                                                                            self.EPOCHS,
                                                                                                            boundary_loss.item(),
                                                                                                            initial_loss.item(),
                                                                                                            common_loss.item()))

        over_train_time = time.time()
        logger.info("train over ...")
        logger.info("train {} epochs in {:.5f}s".format(self.EPOCHS, over_train_time-begin_train_time))

        torch.save(self.pinn_model.state_dict(), self.model_output_path)
        logger.info("save model as {}".format(self.model_output_path))


if __name__ == "__main__":
    _conf = yaml.load(open("./conf/pinn_shrodinger_equation.yaml"), Loader=yaml.FullLoader)
    main_ = ShrodingerEquationPinn(_conf)
    main_.train()
