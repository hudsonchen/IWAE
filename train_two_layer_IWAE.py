# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:44:10 2020

@author: DELL
"""

import numpy as np
import iwae_2
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import os

'''------------------------Hyper-parameter------------------------'''
batch_size = 20
batch_num = 2000
stoc_dim_1 = 100
stoc_dim_2 = 50
det_dim_1 = 200
det_dim_2 = 100
K = 1
feature_size = 28 * 28
epoch_num = 50
eps = 1e-4

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
'''------------------------Loading training data------------------------'''
train_image = np.load("train_data.npy")[:batch_size * batch_num,:].reshape([-1, batch_size, feature_size])

'''------------------------Construct model------------------------'''

model = iwae_2.importance_IWAE_2(feature_size, batch_size, stoc_dim_1, stoc_dim_2, det_dim_1, det_dim_2)
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
		(h1, q_stoc_mean_h1, q_stoc_logvar_h1), (h2, q_stoc_mean_h2, q_stoc_logvar_h2) = model.encode(train_batch_tensor, K, device)
		log_q_hGx = model.get_q_hGx(h1, q_stoc_mean_h1, q_stoc_logvar_h1, K)
		log_q_h = model.get_q_h(h2, q_stoc_mean_h2, q_stoc_logvar_h2, K)

		p_stoc_mean_h1, p_stoc_logvar_h1, X_hat = model.decode(h1, h2)
		log_p_h = model.get_p_h(h1, h2, p_stoc_mean_h1, p_stoc_logvar_h1, K)
		log_p_xGh = model.get_p_xGh(train_batch_tensor, X_hat, K)

		IWAE_ELBO = model.IWAE_loss(log_p_h, log_p_xGh, log_q_hGx, log_q_h)
		optimizer.zero_grad()
		IWAE_ELBO.backward()
		optimizer.step()
	#print(log_p_xGh)
	print('Train epoch: {} IWAE_ELBO: {}'.format((epoch + 1),-IWAE_ELBO.detach()))

model.eval()
test_size = 5000
test_image = np.load("test_data.npy")[:5000,:]

'''------------------------Testing------------------------'''
print("start testing IWAE")

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
(h1, q_stoc_mean_h1_judge, q_stoc_logvar_h1_judge), (h2, q_stoc_mean_h2_judge, q_stoc_logvar_h2_judge) = model.encode(test_tp, 1, device)
q_stoc_mean_h1_judge = q_stoc_mean_h1_judge.detach().cpu().numpy()
q_stoc_mean_h2_judge = q_stoc_mean_h2_judge.detach().cpu().numpy()

judge_matrix_h1 = np.mean(q_stoc_mean_h1_judge ** 2, axis = 0) - np.mean(q_stoc_mean_h1_judge, axis = 0) ** 2
judge_matrix_h2 = np.mean(q_stoc_mean_h2_judge ** 2, axis = 0) - np.mean(q_stoc_mean_h2_judge, axis = 0) ** 2

active_unit_num_h1 = np.sum(judge_matrix_h1 > 0.01)
active_unit_num_h2 = np.sum(judge_matrix_h2 > 0.01)

print('active unit number in layer one: {}'.format(active_unit_num_h1))
print('active unit number in layer two: {}'.format(active_unit_num_h2))

