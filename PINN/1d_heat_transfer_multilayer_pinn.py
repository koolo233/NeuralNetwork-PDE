# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : 1d_heat_transfer_multilayer_pinn.py
# @Time       : 2022/4/5 9:59
import os
import sys

import matplotlib.pyplot as plt

import time
import yaml
import random
import logging
import numpy as np

import torch
from torch import optim

sys.path.append(os.curdir)
from models.PINN_models import DualAccessPINN
from data.DataCreator import OneDHeatMultiLayerDataCreator

# ---------------------- logger ----------------------
logging.basicConfig(level=logging.NOTSET,
                    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create file handler
run_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_file_path = os.path.join("./logs", "{}_{}.log".format(run_time, "1d_heat_transfer_equation_multilayer"))
fh = logging.FileHandler(log_file_path, mode="w")
fh.setLevel(logging.NOTSET)

# output format
basic_format = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(basic_format)

# add logger to handler
logger.addHandler(fh)


class OneDHeatTransferMultiPINN:

    def __init__(self, conf):
        super(OneDHeatTransferMultiPINN, self).__init__()

        logger.info("PINN for {} \n \n \n".format(conf["func_name"]))
        logger.info("hyps list {}".format(conf))
        self.conf = conf

        # ------------------------------------------ #
        # ----------------- global ----------------- #
        # ------------------------------------------ #
        # seed
        random.seed(conf["seed"])
        torch.manual_seed(conf["seed"])
        torch.random.manual_seed(conf["seed"])
        np.random.seed(conf["seed"])

        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # material properties
        self.material1_properties = conf["material1_properties"]
        self.material2_properties = conf["material2_properties"]

        self.material1_properties.append(self.material1_properties[1]/(self.material1_properties[0] *
                                                                       self.material1_properties[2]))
        self.material2_properties.append(self.material2_properties[1]/(self.material2_properties[0] *
                                                                       self.material2_properties[2]))

        # ---------------------------------------- #
        # ----------------- data ----------------- #
        # ---------------------------------------- #

        # sampling
        data_creator = OneDHeatMultiLayerDataCreator(conf)

        sampling_data = data_creator.sampling()

        # raw data
        boundary_data = sampling_data["boundary_data"]
        pde_data = sampling_data["pde_data"]
        initial_data = sampling_data["initial_data"]
        interface_data = sampling_data["interface_data"]

        # ----------------- boundary data -----------------
        self.boundary_x = torch.tensor(boundary_data[0].reshape((-1, 1)), requires_grad=True).float().to(self.device)
        self.boundary_t = torch.tensor(boundary_data[1].reshape((-1, 1))/self.conf["time_total"], requires_grad=True).float().to(self.device)
        # 为了监督边界位置的预测结果，需要实时环境温度数据
        # 假设环境温度如下变化
        # T_0: 293K
        # T_h: 453K
        # 温度上升段: 0~52min
        # 温度保持段：52~172min
        # 温度下降段：172min~222min
        boundary_env_temp = self._compute_exact_env_temperature(boundary_data[1].reshape(-1,))
        self.boundary_env_temp = torch.tensor(boundary_env_temp.reshape((-1, 1))/500.,
                                              requires_grad=True).float().to(self.device)

        # ----------------- PDE -----------------------
        self.pde_x = torch.tensor(pde_data[0].reshape((-1, 1)), requires_grad=True).float().to(self.device)
        self.pde_t = torch.tensor(pde_data[1].reshape((-1, 1))/self.conf["time_total"], requires_grad=True).float().to(self.device)
        # 为了监督PDE函数，需要对应的导热率数据
        heat_transfer_coeff = self._compute_heat_transfer_coeff(pde_data[1])
        self.heat_transfer_coeff = torch.tensor(heat_transfer_coeff.reshape((-1, 1)),
                                                requires_grad=False).float().to(self.device)

        # ----------------- initial condition ---------------------
        self.initial_x = torch.tensor(initial_data[0].reshape((-1, 1)), requires_grad=True).float().to(self.device)
        self.initial_t = torch.tensor(initial_data[1].reshape((-1, 1))/self.conf["time_total"], requires_grad=True).float().to(self.device)
        # 为了监督初始条件，需要初始状态下的环境温度
        self.initial_exact = torch.ones_like(self.initial_x, requires_grad=False) * 293. / 500.
        self.initial_exact = self.initial_exact.reshape((-1, 1)).float().to(self.device)

        # ----------------- interface condition ---------------------
        self.interface_x = torch.tensor(interface_data[0].reshape((-1, 1)), requires_grad=True).float().to(self.device)
        self.interface_t = torch.tensor(interface_data[1].reshape((-1, 1))/self.conf["time_total"], requires_grad=True).float().to(self.device)

        # ----------------------------------------- #
        # ----------------- model ----------------- #
        # ----------------------------------------- #
        self.pinn_model = DualAccessPINN(input_dim=conf["input_dims"],
                                         output_dim=conf["output_dims"],
                                         dim_list=conf["model_layers"],
                                         interface=conf["length_list"][0]).to(self.device)

        logger.info("create pinn done ...")
        logger.info(self.pinn_model)
        # if conf["load_weight"] == "True":
        #     logger.info("load weight ...")
        #     self.pinn_model.load_state_dict(torch.load(conf[]))

        # ------------------------------------------------------ #
        # ----------------- optimizer and loss ----------------- #
        # ------------------------------------------------------ #
        self.optimizer = optim.Adam(self.pinn_model.parameters(), lr=5e-3)
        logger.info("create optimizer and criterion done ...")
        self.iter = 0
        self.max_adam_epochs = conf["max_epochs"]
        # self.optimizer = optim.LBFGS(self.pinn_model.parameters(), lr=conf["lr"],
        #                              max_iter=conf["max_iter"],
        #                              max_eval=conf["max_eval"],
        #                              history_size=conf["history_size"],
        #                              tolerance_grad=eval(conf["tolerance_grad"]),
        #                              tolerance_change=1.0 * np.finfo(float).eps,
        #                              line_search_fn=conf["line_search_fn"])
        # logger.info("create optimizer and criterion done ...")
        # self.iter = 0
        # self.max_adam_epochs = conf["max_epochs"]
        # self.adam_ = False

        # ---------------------------------------- #
        # ----------------- save ----------------- #
        # ---------------------------------------- #
        if not os.path.exists(conf["model_output_root"]):
            os.makedirs(conf["model_output_root"])
        self.model_output_path = os.path.join(conf["model_output_root"], conf["model_output_name"])
        self.pinn_val_output_figure = os.path.join(conf["figure_output_root"], conf["pinn_figure_output_name"])

    def train(self):
        # logger.info("begin train ...")
        # begin_train_time = time.time()
        #
        # self.pinn_model.train()
        #
        # for i in range(1):
        #
        #     logger.info("train with L-BFGS")
        #     self.adam_ = False
        #     self.optimizer.step(self.loss_func)
        #     logger.info("L-BFGS optimize done ...")
        #
        #     adam_optimier = optim.Adam(self.pinn_model.parameters(), lr=2e-3 / np.power(5, i))
        #     logger.info("train with Adam optimizer")
        #     self.adam_ = True
        #     for epoch in range(1, self.max_adam_epochs + 1):
        #         adam_optimier.zero_grad()
        #         self.loss_func()
        #         adam_optimier.step()
        #     logger.info("Adam optimize done ...")
        #
        #     logger.info("train with L-BFGS")
        #     self.adam_ = False
        #     self.optimizer.step(self.loss_func)
        #     logger.info("L-BFGS optimize done ...")
        #
        # over_train_time = time.time()
        # logger.info("train over ...")
        # logger.info("using {:.5f}s".format(self.iter, over_train_time - begin_train_time))
        #
        # torch.save(self.pinn_model.state_dict(), self.model_output_path)
        # logger.info("save model as {}".format(self.model_output_path))
        logger.info("begin train...")
        logger.info("train with Adam optimizer")
        begin_train_time = time.time()
        self.pinn_model.train()

        for epoch in range(1, self.max_adam_epochs + 1):
            self.optimizer.zero_grad()
            self.loss_func()
            self.optimizer.step()

            if epoch == 12000:
                boundary_pred, _, _ = self.pinn_model(self.boundary_x, self.boundary_t/self.conf["time_total"])
                plt.scatter(self.boundary_t.detach().cpu().numpy(), boundary_pred.detach().cpu().numpy())
                plt.show()
        logger.info("optimize done ...")

        over_train_time = time.time()
        logger.info("train over ...")
        logger.info("using {:.5f}s".format(self.iter, over_train_time - begin_train_time))

        torch.save(self.pinn_model.state_dict(), self.model_output_path)
        logger.info("save model as {}".format(self.model_output_path))

    def loss_func(self):
        # ----------------------- PDE损失 -----------------------
        pde_pred, _, _ = self.pinn_model(self.pde_x, self.pde_t)
        # df_dt
        df_dt = torch.autograd.grad(pde_pred, self.pde_t,
                                    grad_outputs=torch.ones_like(pde_pred),
                                    retain_graph=True, create_graph=True)[0]
        # df_dx
        df_dx = torch.autograd.grad(pde_pred, self.pde_x,
                                    grad_outputs=torch.ones_like(pde_pred),
                                    retain_graph=True, create_graph=True)[0]
        # df_dxx
        df_dxx = torch.autograd.grad(df_dx, self.pde_x,
                                     grad_outputs=torch.ones_like(pde_pred),
                                     retain_graph=True, create_graph=True)[0]

        # pde loss
        pde_output = df_dt - self.heat_transfer_coeff * df_dxx
        loss_pde = torch.mean(pde_output ** 2)

        # ----------------------- initial condition 损失 -----------------------
        initial_pred, _, _ = self.pinn_model(self.initial_x, self.initial_t)
        # print(torch.min(initial_pred), torch.max(initial_pred), torch.min(self.initial_exact))
        loss_initial_cond = torch.mean((initial_pred - self.initial_exact) ** 2)

        # ----------------------- boundary condition 损失 ------------------------
        boundary_pred, _, _ = self.pinn_model(self.boundary_x, self.boundary_t)

        # print(torch.min(boundary_pred).item(), torch.max(boundary_pred).item(), torch.mean(boundary_pred).item(),
        #       torch.max(self.boundary_env_temp).item(), torch.min(self.boundary_env_temp).item())

        loss_boundary_cond = None
        if self.conf["boundary_condition_type"] == "convective":
            pass
            # # df_dx
            # df_dx_boundary = torch.autograd.grad(boundary_pred, self.boundary_x,
            #                                      grad_outputs=torch.ones_like(boundary_pred),
            #                                      retain_graph=True, create_graph=True)[0]
            #
        else:
            loss_boundary_cond = torch.mean((boundary_pred - self.boundary_env_temp) ** 2)

        # ----------------------- interface condition 损失 ---------------------------
        _, interface_pred_1, interface_pred_2 = self.pinn_model(self.interface_x, self.interface_t)

        # df_dx
        df_dx_1 = torch.autograd.grad(interface_pred_1, self.interface_x,
                                      grad_outputs=torch.ones_like(interface_pred_1),
                                      retain_graph=True, create_graph=True)[0]
        df_dx_2 = torch.autograd.grad(interface_pred_2, self.interface_x,
                                      grad_outputs=torch.ones_like(interface_pred_2),
                                      retain_graph=True, create_graph=True)[0]

        loss_interface_value = torch.mean((interface_pred_1 - interface_pred_2)**2)
        loss_interface_energy = torch.mean((self.material1_properties[1] * df_dx_1 -
                                            self.material2_properties[1] * df_dx_2)**2)

        total_loss = loss_pde + \
                     200 * loss_boundary_cond + \
                     200 * loss_initial_cond + loss_interface_value + loss_interface_energy
        # total_loss = loss_boundary_cond + loss_initial_cond

        total_loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            # if self.iter == 200:
            #     for param_group in self.optimizer.param_groups:  # 在每次更新参数前迭代更改学习率
            #         param_group["lr"] = 1.0
            logger.info("iter: {}".format(self.iter))
            logger.info("total loss:{:.8e}".format(total_loss.item()))
            logger.info("pde loss:{:.5e}".format(loss_pde.item()))
            logger.info("boundary conditions loss:{:.5e}".format(loss_boundary_cond.item()))
            logger.info("initial conditions loss:{:.5e}".format(loss_initial_cond.item()))
            logger.info("interface condition(value) loss:{:.5e}".format(loss_interface_value.item()))
            logger.info("interface condition(energy) loss:{:.5e}".format(loss_interface_energy.item()))
        return total_loss

    @staticmethod
    def _compute_exact_env_temperature(input_t):

        """
        计算精确环境温度
        :param input_t: 待计算环境温度的时间list
        :return: 与输入时间相对应的环境温度
        """
        # temp_1 = 293
        # temp_2 = 453
        temp_1 = 293
        temp_2 = 453
        critical_point_1 = 52 * 60
        critical_point_2 = 172 * 60
        critical_point_3 = 222 * 60

        exact_temp_list = np.zeros_like(input_t).reshape(-1)

        warmup_stage_point_index_list = np.argwhere(input_t <= critical_point_1)
        keep_stage_point_index_list = np.argwhere((critical_point_1 < input_t) & (input_t <= critical_point_2))
        cooldown_stage_point_index_list = np.argwhere(input_t > critical_point_2)

        # stage 1
        warmup_stage_exact_temp = temp_1 + ((temp_2 - temp_1) / critical_point_1) * \
                                  input_t[warmup_stage_point_index_list]
        exact_temp_list[warmup_stage_point_index_list] = warmup_stage_exact_temp

        # stage 2
        exact_temp_list[keep_stage_point_index_list] = temp_2

        # stage 3
        cooldown_stage_exact_temp = temp_2 + ((temp_1 - temp_2) / (critical_point_3 - critical_point_2)) * \
                                    (input_t[cooldown_stage_point_index_list] - critical_point_2)
        exact_temp_list[cooldown_stage_point_index_list] = cooldown_stage_exact_temp

        return exact_temp_list

    def _compute_heat_transfer_coeff(self, input_x):

        material2_point_index = np.argwhere(input_x > self.conf["length_list"][0])

        coeff_list = np.ones_like(input_x).reshape(-1) * self.material1_properties[-1]
        coeff_list[material2_point_index] = self.material2_properties[-1]

        return coeff_list


if __name__ == "__main__":
    _conf = yaml.load(open("./conf/PINN_1d_heat_transfer_equation_multilayer.yaml"), Loader=yaml.FullLoader)

    main_ = OneDHeatTransferMultiPINN(_conf)
    if _conf["train_model"] == "True":
        main_.train()
    # pred_matrix_ = main_.pred()
    # main_.valuation_(pred_matrix_)
