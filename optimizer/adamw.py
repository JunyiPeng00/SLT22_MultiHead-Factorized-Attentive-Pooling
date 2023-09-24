#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(parameters, lr, **kwargs):

	print('Initialised Adam optimizer')

	return torch.optim.AdamW(parameters, lr = lr);
