# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : PINN_models.py
# @Time       : 2021/12/4 21:03

import torch
from torch import nn


class PINN(nn.Module):

    def __init__(self, input_dim, output_dim, dim_list):
        super(PINN, self).__init__()

        self.pinn_network = nn.Sequential()

        cur_dim = input_dim
        for layer_id, dim in enumerate(dim_list):
            self.pinn_network.add_module("layer_{}".format(layer_id+1), nn.Linear(cur_dim, dim))
            self.pinn_network.add_module("activate_func_{}".format(layer_id+1), nn.Tanh())
            cur_dim = dim
        self.pinn_network.add_module("output_layer", nn.Linear(cur_dim, output_dim))

    def forward(self, x):
        return self.pinn_network(x)


if __name__ == "__main__":
    test_batch_size = 10
    test_input_dim = 2
    test_output_dim = 2

    test_input_tensor = torch.randn((test_batch_size, test_input_dim))
    test_output_tensor = torch.randn((test_batch_size, test_output_dim))
    test_model = PINN(test_input_dim, test_output_dim, [100, 100, 100, 100])

    pred_ = test_model(test_input_tensor)
    print(test_model)
    print(pred_.shape, test_output_tensor.shape)
