# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : Burgers_equation_pinn.py
# @Time       : 2021/12/6 22:04
import os
import sys
sys.path.append(os.curdir)

import time
import yaml
import random
import logging
import numpy as np

import torch
from torch import optim

from models.PINN_models import PINN
from data.DataCreator import BurgersEquationDataCreator

# ---------------------- logger ----------------------
logging.basicConfig(level=logging.NOTSET,
                    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
# create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create file handler
run_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_file_path = os.path.join("./logs", "{}_{}.log".format(run_time, "burgers_equation"))
fh = logging.FileHandler(log_file_path, mode="w")
fh.setLevel(logging.NOTSET)

# output format
basic_format = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(basic_format)

# add logger to handler
logger.addHandler(fh)


class BurgersEquationPINN(object):

    def __init__(self, conf, load_weight_path=None):
        super(BurgersEquationPINN, self).__init__()

        logger.info("PINN for Burgers Equation \n \n \n")
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
        self.data_creator = BurgersEquationDataCreator(conf)
        begin_fd_time = time.time()
        logger.info("create data ...")
        self.data_creator.iter_func()
        over_fd_time = time.time()
        logger.info(
            "finite difference with fourth lunge-kutta method uses {:.5f}s".format(over_fd_time - begin_fd_time)
        )
        # plot numerical data
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
        layers = [20, 20, 20, 20, 20, 20, 20, 20]
        self.pinn_model = PINN(input_dim=2, output_dim=1, dim_list=layers).to(self.device)
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

        # ---------------------------------------- #
        # ----------------- save ----------------- #
        # ---------------------------------------- #
        self.model_output_path = os.path.join(conf["model_output_root"], conf["model_output_name"])
        self.pinn_val_output_figure = os.path.join(conf["figure_output_root"], conf["pinn_figure_output_name"])

    def train(self):
        logger.info("begin train ...")
        begin_train_time = time.time()

        self.pinn_model.train()
        self.optimizer.step(self.loss_func)

        over_train_time = time.time()
        logger.info("train over ...")
        logger.info("using {:.5f}s".format(self.iter, over_train_time-begin_train_time))

        torch.save(self.pinn_model.state_dict(), self.model_output_path)
        logger.info("save model as {}".format(self.model_output_path))

    def loss_func(self):
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

        pde_output = df_dt + f_pred * df_dx - (0.01 / np.pi) * df_dxx

        loss_u = torch.mean((u_pred - self.u_exact) ** 2)
        loss_f = torch.mean(pde_output ** 2)

        total_loss = loss_f + loss_u

        total_loss.backward()

        self.iter += 1
        if self.iter % 100 == 0:
            logger.info("iter: {}".format(self.iter))
            logger.info("total loss:{:.8e}".format(total_loss.item()))
            logger.info("boundary and initial conditions loss:{:.5e}\t "
                        "pde function loss:{:.5e}".format(loss_u.item(), loss_f.item()))
        return total_loss

    def pred_and_valuation(self):
        logger.info("begin pred ...")
        with torch.no_grad():
            self.pinn_model.eval()

            # pred output init
            true_output_matrix = self.data_creator.result_matrix
            x_position_list = np.linspace(self.data_creator.cal_x_range[0], self.data_creator.cal_x_range[1],
                                          self.data_creator.space_n).reshape((-1, 1))
            t_position_list = np.linspace(0, self.data_creator.time_total, self.data_creator.time_n)

            pred_output_matrix = np.zeros((int(self.data_creator.space_n),
                                           int(self.data_creator.time_n)))

            x_position_tensor = torch.from_numpy(x_position_list).type(torch.float32).to(self.device)
            # pred
            begin_pred_time = time.time()
            for t, time_point in enumerate(t_position_list):

                # create input t tensor
                t_tensor = torch.ones_like(x_position_tensor) * time_point

                pred_ = self.pinn_model(x_position_tensor, t_tensor.to(self.device)).detach().cpu().numpy()

                pred_output_matrix[:, t] = np.reshape(pred_.copy(), (-1))

                if t % 1000 == 0:
                    print(np.max(pred_output_matrix[:, t]), np.min(pred_output_matrix[:, t]))
                    logger.info("[{}/{}] pred done ...".format(t, len(t_position_list)))
            over_pred_time = time.time()
            logger.info("pred done ...")
            logger.info("using {:.5f}s ...".format(over_pred_time - begin_pred_time))

        # valuation
        l1_norm = np.linalg.norm(pred_output_matrix.reshape(-1) - true_output_matrix.reshape(-1), 1)
        l2_norm = np.linalg.norm(pred_output_matrix.reshape(-1) - true_output_matrix.reshape(-1))

        max_error = np.max(np.abs(pred_output_matrix - true_output_matrix))
        min_error = np.min(np.abs(pred_output_matrix - true_output_matrix))
        average_error = np.average(np.abs(pred_output_matrix - true_output_matrix))

        logger.info("l1 norm {:.7f}".format(l1_norm))
        logger.info("l2 norm {:.7f}".format(l2_norm))
        logger.info("max error {:.7f}".format(max_error))
        logger.info("min error {:.7f}".format(min_error))
        logger.info("average error {:.7f}".format(average_error))

        self.data_creator.figure_output_path = os.path.join(self.conf["figure_output_root"],
                                                            self.conf["pinn_figure_output_name"])
        self.data_creator.gif_output_path = os.path.join(self.conf["figure_output_root"],
                                                         self.conf["pinn_gif_output_name"])
        self.data_creator.result_matrix = pred_output_matrix.copy()
        self.data_creator.plot_func(save_figure=True)


if __name__ == "__main__":
    _conf = yaml.load(open("./conf/pinn_burgers_equation.yaml"), Loader=yaml.FullLoader)
    weight_path = r"./output/weights/burgers_equation_pinn.pth"
    main_ = BurgersEquationPINN(_conf, weight_path)
    # main_.train()
    main_.pred_and_valuation()
