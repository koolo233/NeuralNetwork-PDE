# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : MyDataset.py
# @Time       : 2021/12/4 21:04
import time

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, data_list, dims=2, device=None) -> None:
        """
        sample_i: [x_1, x_2, ..., x_{n-1}, t, true_output_real, true_output_imag]
        :param data_list: [sample_1, sample_2, ..., sample_n]
        :param dims: data dims, default: (x, t), if dims=n, (x_1, x_2, ..., x_{n-1}, t)
        """
        super(MyDataset, self).__init__()
        self.data_list = data_list
        self.dims = dims
        if device is not None:
            self.device = device

    def __getitem__(self, item):

        input_vector = self.data_list[item][:self.dims]
        output_vector = self.data_list[item][self.dims:]

        return {"input_vector": input_vector, "output_vector": output_vector}

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):

        input_list = list()
        output_list = list()
        for sample in batch:
            input_list.append(torch.unsqueeze(torch.from_numpy(sample["input_vector"]), 0))
            output_list.append(torch.unsqueeze(torch.from_numpy(sample["output_vector"]), 0))

        input_tensor = torch.cat(input_list, dim=0).type(torch.float32)
        input_tensor_list = [input_tensor[:, i].requires_grad_() for i in range(input_tensor.shape[-1])]
        output_tensor = torch.cat(output_list, dim=0).type(torch.float32)

        return {"input": input_tensor_list, "output": output_tensor}


if __name__ == "__main__":
    import numpy as np
    from torch.utils.data import DataLoader
    test_data = np.zeros((100, 4))
    test_data[:, 2:] = 1

    test_dataset = MyDataset(test_data, dims=2)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=test_dataset.collate_fn)
    for i, test_batch in enumerate(test_dataloader):
        test_input_tensor = test_batch["input"]
        test_output_tensor = test_batch["output"]
        for j in range(len(test_input_tensor)):
            print(test_input_tensor[j].shape)
        print(torch.sum(test_output_tensor))
