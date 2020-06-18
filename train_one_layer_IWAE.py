# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:44:10 2020

@author: DELL
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import iwae
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

'''------------------------Hyper-parameter------------------------'''
batch_size = 20
batch_num = 2000
stoc_dim = 50
det_dim = 200
K = 50
feature_size = 28 * 28
epoch_num = 30
eps = 1e-4



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
'''------------------------Loading training data------------------------'''
train_image = np.load("train_data.npy")[:batch_size * batch_num,:].reshape([-1, batch_size, feature_size])

'''------------------------Construct model------------------------'''

model = iwae.importance_vae(feature_size, stoc_dim, det_dim)
model.to(device)

'''------------------------Optimizer------------------------'''
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5,gamma = 0.9)

'''------------------------Training IWAE------------------------'''
print("start training IWAE")
for epoch in range(epoch_num):
	scheduler.step()
	for batch_idx in range(batch_num):
		train_batch_tensor = Variable(torch.Tensor(train_image[batch_idx,:,:])).to(device)
		stoc_mean, stoc_logvar = model.encode(train_batch_tensor)
		K_samples = model.get_K_samples(stoc_mean, stoc_logvar, K, device)
		X_hat = model.decode(K_samples)
		
		IWAE_ELBO = model.IWAE_loss(train_batch_tensor, X_hat, K_samples, batch_size, K, stoc_mean, stoc_logvar,  feature_size, stoc_dim, device)
		optimizer.zero_grad()
		IWAE_ELBO.backward()
		optimizer.step()
	print('Train epoch: {} IWAE_ELBO: {}'.format((epoch + 1),-IWAE_ELBO.detach()))

model.eval()
test_size = 5000
test_image = np.load("test_data.npy")[:5000,:]

'''------------------------Testing------------------------'''
print("start testing IWAE")
"""
for i in range(10):
	test_tensor = torch.Tensor(test_image[i,:]).to(device)
	test_hat = model.forward(test_tensor, 1, device).reshape([28,28])
	if (i == 0):
		idiot = torch.cat((test_tensor.reshape([28,28]),test_hat), dim = 0)
		output = idiot
	else:
		idiot = torch.cat((test_tensor.reshape([28,28]),test_hat), dim = 0)
		output = torch.cat((output, idiot), dim = 1)
save_image(output, 'test_result.png')
"""
test_tensor = torch.Tensor(test_image).to(device)
test_hat = model.forward(test_tensor, 1, device)
test_NLL = torch.mean(torch.sum(test_tensor * torch.log(test_hat + eps) + (1. - test_tensor) * torch.log(1. - test_hat + eps), dim = 1))
print('NLL: {}'.format(test_NLL))

"""
'''------------------------Faking------------------------'''
for i in range(10):
	Alzheimer = model.generate_random_latent(stoc_dim).reshape([28,28])
	if (i == 0):
		fake_image = Alzheimer
	else:
		fake_image = torch.cat((fake_image, Alzheimer), dim = 1)	
save_image(fake_image, 'fake_result.png')
"""

'''------------------------Judge active units------------------------'''
test_tp = test_tensor[:1000,:]
stoc_mean_judgeactive, stoc_logvar_judgeactive = model.encode(test_tp)
stoc_mean_judgeactive = stoc_mean_judgeactive.detach().cpu().numpy()
judge_matrix = np.mean(stoc_mean_judgeactive ** 2, axis = 0) - np.mean(stoc_mean_judgeactive, axis = 0) ** 2
active_unit_num = np.sum(judge_matrix > 0.01)
print('active unit number: {}'.format(active_unit_num))

