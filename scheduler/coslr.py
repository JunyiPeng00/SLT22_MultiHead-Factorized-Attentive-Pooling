#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, eta_min=6e-6, T_max=8)
	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step
