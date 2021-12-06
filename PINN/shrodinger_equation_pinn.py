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

import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.utils.data import DataLoader

from data.MyDataset import MyDataset
from models.PINN_models import PINN
from data.DataCreator import ShrodingerEquationDataCreator


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

# output format
basic_format = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(basic_format)

# add logger to handler
logger.addHandler(fh)


class ShrodingerEquationPinn(object):
    
    def __init__(self, conf, load_weight_path=None):
        super(ShrodingerEquationPinn, self).__init__()
        logger.info("PINN for Shrodinger Equation \n \n \n")
        logger.info("hyps list {}".format(conf))
        self.conf = conf

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
        self.data_creator = ShrodingerEquationDataCreator(conf)
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
        if conf["load_weight"] == "True":
            logger.info("load weight ...")
            self.pinn_model.load_state_dict(torch.load(load_weight_path))

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
                logger.info("[{}/{}]\tboundary loss:{:.7f}\tinitial loss:{:.7f}\tcommon loss:{:.7f}".format(epoch,
                                                                                                            self.EPOCHS,
                                                                                                            boundary_loss.item(),
                                                                                                            initial_loss.item(),
                                                                                                            common_loss.item()))

        over_train_time = time.time()
        logger.info("train over ...")
        logger.info("train {} epochs in {:.5f}s".format(self.EPOCHS, over_train_time-begin_train_time))

        torch.save(self.pinn_model.state_dict(), self.model_output_path)
        logger.info("save model as {}".format(self.model_output_path))

    def pred_and_valuation(self):
        logger.info("begin pred ...")
        with torch.no_grad():
            self.pinn_model.eval()

            # data
            true_output_matrix = self.data_creator.phi_matrix
            x_position_list = np.linspace(-self.data_creator.box_l / 2, self.data_creator.box_l / 2,
                                          self.data_creator.space_n)
            t_position_list = np.linspace(0, self.data_creator.time_total, self.data_creator.time_n)

            pred_output_matrix = np.zeros((int(self.data_creator.space_n),
                                           int(self.data_creator.time_n))).astype(np.complex64)

            x_position_tensor = torch.from_numpy(x_position_list).type(torch.float32).to(self.device)

            # pred
            begin_pred_time = time.time()
            for t, time_point in enumerate(t_position_list):
                t_tensor = torch.ones_like(x_position_tensor)*time_point

                pred_ = self.pinn_model(x_position_tensor, t_tensor.to(self.device)).detach().cpu().numpy()

                pred_output_matrix[:, t] = pred_[:, 0] + 1.j * pred_[:, 1]

                if t % 1000 == 0:
                    print(np.max(pred_output_matrix[:, t]), np.min(pred_output_matrix[:, t]))
                    logger.info("[{}/{}] pred done ...".format(t, len(t_position_list)))
            over_pred_time = time.time()
            logger.info("pred done ...")
            logger.info("pred over at {:.5f}s ...".format(over_pred_time-begin_pred_time))

        # valuation
        l1_norm = np.linalg.norm(pred_output_matrix.reshape(-1)-true_output_matrix.reshape(-1), 1)
        l2_norm = np.linalg.norm(pred_output_matrix.reshape(-1)-true_output_matrix.reshape(-1))

        max_error = np.max(np.abs(pred_output_matrix-true_output_matrix))
        min_error = np.min(np.abs(pred_output_matrix-true_output_matrix))
        average_error = np.average(np.abs(pred_output_matrix-true_output_matrix))

        logger.info("l1 norm {:.7f}".format(l1_norm))
        logger.info("l2 norm {:.7f}".format(l2_norm))
        logger.info("max error {:.7f}".format(max_error))
        logger.info("min error {:.7f}".format(min_error))
        logger.info("average error {:.7f}".format(average_error))

        self.data_creator.figure_output_path = os.path.join(self.conf["figure_output_root"],
                                                            self.conf["pinn_figure_output_name"])
        self.data_creator.phi_matrix = pred_output_matrix.copy()
        self.data_creator.plot_func(save_figure=True)


if __name__ == "__main__":
    _conf = yaml.load(open("./conf/pinn_shrodinger_equation.yaml"), Loader=yaml.FullLoader)
    weight_path = r"./output/weights/shrodinger_equation_pinn.pth"
    main_ = ShrodingerEquationPinn(_conf, weight_path)
    main_.train()
    main_.pred_and_valuation()
