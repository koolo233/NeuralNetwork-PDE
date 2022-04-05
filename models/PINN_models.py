# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : PINN_models.py
# @Time       : 2021/12/4 21:03

import torch
from torch import nn
import torch.nn.functional as F


class BasicPINN(nn.Module):

    def __init__(self, input_dim, output_dim, dim_list):
        super(BasicPINN, self).__init__()

        self.pinn_network = nn.Sequential()

        cur_dim = input_dim
        for layer_id, dim in enumerate(dim_list):
            self.pinn_network.add_module("layer_{}".format(layer_id+1), nn.Linear(cur_dim, dim))
            self.pinn_network.add_module("activate_func_{}".format(layer_id+1), nn.Tanh())
            cur_dim = dim
        self.pinn_network.add_module("output_layer", nn.Linear(cur_dim, output_dim))

    def forward(self, x):

        return self.pinn_network(x)


class PINN(nn.Module):

    def __init__(self, input_dim, output_dim, dim_list):
        super(PINN, self).__init__()

        self.model = BasicPINN(input_dim, output_dim, dim_list)

    def forward(self, x, t):

        # x = torch.unsqueeze(x, -1)
        # t = torch.unsqueeze(t, -1)
        input_tensor = torch.cat((x, t), -1)

        return self.model(input_tensor)


class DualAccessPINN(nn.Module):

    def __init__(self, input_dim, output_dim, dim_list, interface):
        super(DualAccessPINN, self).__init__()

        assert len(dim_list) == 2

        self.model_1 = BasicPINN(input_dim, output_dim, dim_list[0])
        self.model_2 = BasicPINN(input_dim, output_dim, dim_list[1])

        self.softplus = F.softplus

        self.heaviside_func = MyHeaviside()

        self.interface = interface

    def forward(self, x, t):

        input_tensor = torch.cat((x, t), -1)

        output_1 = self.softplus(self.model_1(input_tensor))
        output_2 = self.softplus(self.model_2(input_tensor))

        # mask_1 = self.heaviside_func.apply(self.interface - x)
        # mask_2 = -mask_1 + 1
        #
        # output_tensor = mask_1 * output_1 + mask_2 * output_2
        output_tensor = output_1

        return output_tensor, output_1, output_2


class MyHeavisidePlus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, lower_value, larger_value):

        _mask = (x > 0).to(torch.float32) * (larger_value - lower_value) + lower_value

        ctx.save_for_backward(_mask)
        return _mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 0


class MyHeaviside(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        judge_result = (x > 0).to(torch.float32)

        ctx.save_for_backward(judge_result)
        return judge_result

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 0


if __name__ == "__main__":
    test_batch_size = 10
    test_input_dim = 2
    test_output_dim = 1

    test_input_tensor = torch.randn((test_batch_size, test_input_dim))
    test_output_tensor = torch.randn((test_batch_size, test_output_dim))

    test_model = DualAccessPINN(test_input_dim, test_output_dim,
                                [[100, 100, 100, 100], [100, 100, 100, 100]], 0)

    pred_ = test_model(test_input_tensor[:, 0:1], test_input_tensor[:, 1:])
    print(pred_)

    my_heaviside_func = MyHeaviside()

    print(torch.autograd.gradcheck(lambda x: my_heaviside_func.apply(x), torch.randn(10, requires_grad=True)))

    # test_model = PINN(test_input_dim, test_output_dim, [100, 100, 100, 100])
    #
    # pred_ = test_model(test_input_tensor)
    # print(test_model)
    # print(pred_.shape, test_output_tensor.shape)
