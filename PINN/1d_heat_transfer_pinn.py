# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : 1d_heat_transfer_pinn.py
# @Time       : 2021/12/6 21:48
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.curdir)

import time
import yaml
import random
import logging
import numpy as np

import torch
from torch import optim

from models.PINN_models import PINN
from data.DataCreator import OneDHeatTransferEquationDataCreator

# ---------------------- logger ----------------------
logging.basicConfig(level=logging.NOTSET,
                    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create file handler
run_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_file_path = os.path.join("./logs", "{}_{}.log".format(run_time, "1d_heat_transfer_equation"))
fh = logging.FileHandler(log_file_path, mode="w")
fh.setLevel(logging.NOTSET)

# output format
basic_format = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(basic_format)

# add logger to handler
logger.addHandler(fh)


class OneDHeatTransferPINN(object):

    def __init__(self, conf, load_weight_path=None):
        super(OneDHeatTransferPINN, self).__init__()

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

        # ---------------------------------------- #
        # ----------------- data ----------------- #
        # ---------------------------------------- #

        # create numerical data
        self.data_creator = OneDHeatTransferEquationDataCreator(conf)
        begin_fd_time = time.time()
        logger.info("create data ...")
        self.data_creator.iter_func()
        over_fd_time = time.time()
        logger.info(
            "finite difference with fourth lunge-kutta method uses {:.5f}s".format(over_fd_time - begin_fd_time)
        )
        # plot numerical data
        if conf["numerical_plot"] == "True":
            logger.info("create figure and gif ...")
            self.data_creator.plot_func(save_figure=True)
            logger.info("save figure as {}".format(self.data_creator.figure_output_path))
            logger.info("save gif as {}".format(self.data_creator.gif_output_path))

        # sampling
        data_dict = self.data_creator.sampling(boundary_num=conf["boundary_num"],
                                               initial_num=conf["initial_num"],
                                               common_num=conf["common_num"], seed=conf["seed"])

        # raw data
        boundary_data = data_dict["boundary"]
        initial_data = data_dict["initial"]
        common_data = data_dict["common"]

        u_train_x = np.vstack([initial_data[:, 0:2], boundary_data[:, 0:2]])
        u_train_exact = np.vstack([initial_data[:, 2].reshape((-1, 1)), boundary_data[:, 2].reshape((-1, 1))])
        f_train_x = common_data[:, 0:2]

        self.u_x = torch.tensor(u_train_x[:, 0:1], requires_grad=True).float().to(self.device)
        self.u_t = torch.tensor(u_train_x[:, 1:2], requires_grad=True).float().to(self.device)
        self.f_x = torch.tensor(f_train_x[:, 0:1], requires_grad=True).float().to(self.device)
        self.f_t = torch.tensor(f_train_x[:, 1:2], requires_grad=True).float().to(self.device)
        self.u_exact = torch.tensor(u_train_exact).float().to(self.device)

        # ----------------------------------------- #
        # ----------------- model ----------------- #
        # ----------------------------------------- #
        self.pinn_model = PINN(input_dim=2, output_dim=1, dim_list=conf["model_layers"]).to(self.device)
        logger.info("create pinn done ...")
        logger.info(self.pinn_model)
        if conf["load_weight"] == "True":
            logger.info("load weight ...")
            self.pinn_model.load_state_dict(torch.load(load_weight_path))

        # ------------------------------------------------------ #
        # ----------------- optimizer and loss ----------------- #
        # ------------------------------------------------------ #
        self.optimizer = optim.LBFGS(self.pinn_model.parameters(), lr=conf["lr"],
                                     max_iter=conf["max_iter"],
                                     max_eval=conf["max_eval"],
                                     history_size=conf["history_size"],
                                     tolerance_grad=eval(conf["tolerance_grad"]),
                                     tolerance_change=1.0 * np.finfo(float).eps,
                                     line_search_fn=conf["line_search_fn"])
        logger.info("create optimizer and criterion done ...")
        self.iter = 0
        self.max_adam_epochs = conf["max_epochs"]
        self.adam_ = False

        # ---------------------------------------- #
        # ----------------- save ----------------- #
        # ---------------------------------------- #
        if not os.path.exists(conf["model_output_root"]):
            os.makedirs(conf["model_output_root"])
        self.model_output_path = os.path.join(conf["model_output_root"], conf["model_output_name"])
        self.pinn_val_output_figure = os.path.join(conf["figure_output_root"], conf["pinn_figure_output_name"])

    def train(self):
        logger.info("begin train ...")
        begin_train_time = time.time()

        self.pinn_model.train()

        for i in range(1):

            logger.info("train with L-BFGS")
            self.adam_ = False
            self.optimizer.step(self.loss_func)
            logger.info("L-BFGS optimize done ...")

            adam_optimier = optim.Adam(self.pinn_model.parameters(), lr=2e-3/np.power(5, i))
            logger.info("train with Adam optimizer")
            self.adam_ = True
            for epoch in range(1, self.max_adam_epochs + 1):
                adam_optimier.zero_grad()
                self.loss_func()
                adam_optimier.step()
            logger.info("Adam optimize done ...")

            logger.info("train with L-BFGS")
            self.adam_ = False
            self.optimizer.step(self.loss_func)
            logger.info("L-BFGS optimize done ...")

        over_train_time = time.time()
        logger.info("train over ...")
        logger.info("using {:.5f}s".format(self.iter, over_train_time-begin_train_time))

        torch.save(self.pinn_model.state_dict(), self.model_output_path)
        logger.info("save model as {}".format(self.model_output_path))

    def loss_func(self):
        if not self.adam_:
            self.optimizer.zero_grad()

        u_pred = self.pinn_model(self.u_x, self.u_t)
        f_pred = self.pinn_model(self.f_x, self.f_t)

        # du_dx
        df_dx = torch.autograd.grad(f_pred, self.f_x,
                                    grad_outputs=torch.ones_like(f_pred),
                                    retain_graph=True, create_graph=True)[0]
        # du_dt
        df_dt = torch.autograd.grad(f_pred, self.f_t,
                                    grad_outputs=torch.ones_like(f_pred),
                                    retain_graph=True, create_graph=True)[0]

        # du_dxx
        df_dxx = torch.autograd.grad(df_dx, self.f_x,
                                     grad_outputs=torch.ones_like(df_dx),
                                     retain_graph=True, create_graph=True)[0]

        pde_output = df_dt - 0.1 * df_dxx

        loss_u = torch.mean((u_pred - self.u_exact) ** 2)
        loss_f = torch.mean(pde_output ** 2)

        total_loss = loss_f + loss_u

        total_loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            # if self.iter == 200:
            #     for param_group in self.optimizer.param_groups:  # ?????????????????????????????????????????????
            #         param_group["lr"] = 1.0
            logger.info("iter: {}".format(self.iter))
            logger.info("total loss:{:.8e}".format(total_loss.item()))
            logger.info("boundary and initial conditions loss:{:.5e}\t "
                        "pde function loss:{:.5e}".format(loss_u.item(), loss_f.item()))
        return total_loss

    def pred(self):
        logger.info("begin pred ...")

        self.conf["time_total"] = 20
        self.data_creator = OneDHeatTransferEquationDataCreator(self.conf)
        self.data_creator.iter_func()

        with torch.no_grad():
            self.pinn_model.eval()

            # pred output init
            true_output_matrix = self.data_creator.result_matrix
            x_position_list = np.linspace(-self.data_creator.cal_x_range[0], self.data_creator.cal_x_range[1],
                                          self.data_creator.space_n).reshape((-1, 1))
            t_position_list = np.linspace(0, self.data_creator.time_total, self.data_creator.time_n)

            pred_output_matrix = np.zeros((int(self.data_creator.space_n),
                                           int(self.data_creator.time_n)))

            # pred
            begin_pred_time = time.time()

            x_position_input, y_position_input = np.meshgrid(x_position_list, t_position_list)
            x_position_input = torch.from_numpy(x_position_input.reshape(-1, 1)).to(dtype=torch.float32, device=self.device)
            y_position_input = torch.from_numpy(y_position_input.reshape(-1, 1)).to(dtype=torch.float32, device=self.device)
            pred_ = self.pinn_model(x_position_input, y_position_input).detach().cpu().numpy()
            pred_output_matrix = pred_.reshape((len(t_position_list), len(x_position_list))).T


            # for t, time_point in enumerate(t_position_list):
            #
            #     # create input t tensor
            #     t_tensor = torch.ones_like(x_position_tensor) * time_point
            #
            #
            #
            #     pred_output_matrix[:, t] = np.reshape(pred_.copy(), (-1))
            #
            #     if t % 1000 == 0:
            #         print(np.max(pred_output_matrix[:, t]), np.min(pred_output_matrix[:, t]))
            #         logger.info("[{}/{}] pred done ...".format(t, len(t_position_list)))
            over_pred_time = time.time()
            logger.info("pred done ...")
            logger.info("using {:.5f}s ...".format(over_pred_time - begin_pred_time))

        self.data_creator.figure_output_path = os.path.join(self.conf["figure_output_root"],
                                                            self.conf["pinn_figure_output_name"])
        self.data_creator.gif_output_path = os.path.join(self.conf["figure_output_root"],
                                                         self.conf["pinn_gif_output_name"])
        self.data_creator.result_matrix = pred_output_matrix.copy()
        self.data_creator.plot_func(save_figure=True)

        # different figure
        different_figure = np.abs(pred_output_matrix - true_output_matrix)

        self.data_creator.figure_output_path = os.path.join(self.conf["figure_output_root"],
                                                            self.conf["different_figure_output_name"])
        self.data_creator.gif_output_path = os.path.join(self.conf["figure_output_root"],
                                                         self.conf["different_gif_output_name"])
        self.data_creator.result_matrix = different_figure.copy()
        self.data_creator.plot_func(save_figure=True)

        # ????????????
        # ???????????? 0??? 0.25, 0.5, 0.75, 1.5, 5, 10, 15, 19???????????????
        time_points = [0, int(0.25/self.conf["delta_time"]),
                       int(0.5/self.conf["delta_time"]),
                       int(0.75/self.conf["delta_time"]),
                       int(1.5/self.conf["delta_time"]),
                       int(5/self.conf["delta_time"]),
                       int(10/self.conf["delta_time"]),
                       int(15/self.conf["delta_time"]),
                       int(19/self.conf["delta_time"])]

        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.figure(figsize=(10, 10))
        x_value_list = np.linspace(self.conf["cal_x_range"][0],
                                   self.conf["cal_x_range"][1], true_output_matrix.shape[0])
        for _time in time_points:
            true_vector = true_output_matrix[:, _time]
            pred_vector = pred_output_matrix[:, _time]
            plt.clf()
            plt.cla()
            plt.title("t={}".format(np.round(_time * self.conf["delta_time"], 2)))
            plt.xlabel("x")
            plt.ylabel("u(t, x)")

            max_y = np.max(np.hstack((true_vector, pred_vector)))
            min_y = np.min(np.hstack((true_vector, pred_vector)))

            plt.ylim(min_y-(max_y-min_y)*0.05, max_y+(max_y-min_y)*0.05)
            plt.plot(x_value_list, pred_vector, label="PINN????????????", linewidth=2)
            plt.plot(x_value_list, true_vector, label="??????????????????", linewidth=2)
            print(np.average(np.abs(pred_vector-true_vector)))
            plt.legend()
            plt.savefig("./output/plots/1d_heat_transfer_pinn_time_{}.png".format(np.round(_time *
                                                                                           self.conf["delta_time"], 2)))
        return pred_output_matrix

    def valuation_(self, pred_matrix):
        self.data_creator = OneDHeatTransferEquationDataCreator(self.conf)
        self.data_creator.iter_func()

        true_output_matrix = self.data_creator.result_matrix
        # valuation
        l1_norm = np.linalg.norm(pred_matrix.reshape(-1) - true_output_matrix.reshape(-1), 1)
        l2_norm = np.linalg.norm(pred_matrix.reshape(-1) - true_output_matrix.reshape(-1))

        max_error = np.max(np.abs(pred_matrix[int(self.data_creator.time_total/2):, :]
                                  - true_output_matrix[int(self.data_creator.time_total/2):, :]))
        min_error = np.min(np.abs(pred_matrix[int(self.data_creator.time_total/2):, :] -
                                  true_output_matrix[int(self.data_creator.time_total/2):, :]))
        average_error = np.average(np.abs(pred_matrix[int(self.data_creator.time_total/2):, :] -
                                          true_output_matrix[int(self.data_creator.time_total/2):, :]))

        logger.info("l1 norm {:.7f}".format(l1_norm))
        logger.info("l2 norm {:.7f}".format(l2_norm))
        logger.info("max error {:.7f}".format(max_error))
        logger.info("min error {:.7f}".format(min_error))
        logger.info("average error {:.7f}".format(average_error))

        return average_error


if __name__ == "__main__":
    _conf = yaml.load(open("./conf/pinn_1d_heat_transfer_equation.yaml"), Loader=yaml.FullLoader)
    weight_path = r"./output/weights/1d_heat_transfer_equation_pinn.pth"

    main_ = OneDHeatTransferPINN(_conf, weight_path)
    if _conf["train_model"] == "True":
        main_.train()
    pred_matrix_ = main_.pred()
    main_.valuation_(pred_matrix_)

    # layer_num_list = [6]
    # neural_num_list = [40]
    #
    # boundary_initial_num = 200
    # common_num = 16000
    # boundary_num = int(boundary_initial_num * 0.7)
    # initial_num = int(boundary_initial_num * 0.3)
    #
    # for layer_conf in layer_num_list:
    #     for neural_num_conf in neural_num_list:
    #         logger.info("\n \n \n")
    #         logger.info("layer_conf {}".format(layer_conf))
    #         logger.info("neural_num_conf {}".format(neural_num_conf))
    #
    #         _conf["model_layers"] = np.repeat(neural_num_conf, layer_conf)
    #         logger.info("model_layers {}".format(_conf["model_layers"]))
    #
    #         main_ = OneDHeatTransferPINN(_conf, weight_path)
    #         main_.train()
    #         pred_matrix_ = main_.pred()
    #         main_.valuation_(pred_matrix_)


    # common_num_list = [4000, 8000, 12000, 16000, 20000]
    # boundary_initial_num_list = [100, 140, 180, 220]
    #
    # error_matrix = np.zeros((len(common_num_list), len(boundary_initial_num_list)))
    # for i, boundary_initial_num in enumerate(boundary_initial_num_list):
    #     for j, common_num in enumerate(common_num_list):
    #
    #         train_ = False
    #         if (boundary_initial_num == 180 and common_num == 20000) or (boundary_initial_num == 140 and common_num == 12000):
    #             train_ = True
    #         if train_:
    #             boundary_num = int(boundary_initial_num * 0.7)
    #             initial_num = int(boundary_initial_num * 0.3)
    #
    #             logger.info("\n \n \n")
    #             logger.info("common_num {}".format(common_num))
    #             logger.info("initial_num {}".format(initial_num))
    #             logger.info("boundary_num {}".format(boundary_num))
    #
    #             _conf["boundary_num"] = boundary_num
    #             _conf["initial_num"] = initial_num
    #             _conf["common_num"] = common_num
    #
    #             main_ = OneDHeatTransferPINN(_conf, weight_path)
    #             main_.train()
    #             # if _conf["PINN_plot"] == "True":
    #             pred_matrix_ = main_.pred()
    #             error_matrix[j][i] = main_.valuation_(pred_matrix_)
    #
    # logger.info("error matrix".format(error_matrix))
