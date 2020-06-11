# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:44:10 2020

@author: DELL
"""

import numpy as np
import iwae
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

'''------------------------Hyper-parameter------------------------'''
batch_size = 20
batch_num = 1000
stoc_dim = 50
det_dim = 100
K = 5
feature_size = 28 * 28
epoch_num = 40

'''------------------------Loading training data------------------------'''
train_image = np.load("train_data.npy")[:batch_size * batch_num,:].reshape([-1, batch_size, feature_size])

'''------------------------Construct model------------------------'''
model = iwae.importance_vae(feature_size, stoc_dim, det_dim)


'''------------------------Optimizer------------------------'''
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 4,gamma = 0.9)

'''------------------------Training------------------------'''
for epoch in range(epoch_num):
	scheduler.step()
	for batch_idx in range(batch_num):
		train_batch_tensor = Variable(torch.Tensor(train_image[batch_idx,:,:]))
		stoc_mean, stoc_logvar = model.encode(train_batch_tensor)
		K_samples = model.get_K_samples(stoc_mean, stoc_logvar, K)
		X_hat = model.decode(K_samples)
		
		ELBO = model.IWAE_loss(train_batch_tensor, X_hat, K_samples, batch_size, K, stoc_mean, stoc_logvar,  feature_size, stoc_dim)
		optimizer.zero_grad()
		ELBO.backward()
		optimizer.step()
	print('Train epoch: {} ELBO: {}'.format((epoch + 1),-ELBO.detach()))

model.eval()
test_image = np.load("test_data.npy")[:10,:]

'''------------------------Testing------------------------'''
for i in range(10):
	test_tensor = torch.Tensor(test_image[i,:])
	test_hat = model.forward(test_tensor, 1).reshape([28,28])
	if (i == 0):
		idiot = torch.cat((test_tensor.reshape([28,28]),test_hat), dim = 0)
		output = idiot
	else:
		idiot = torch.cat((test_tensor.reshape([28,28]),test_hat), dim = 0)
		output = torch.cat((output, idiot), dim = 1)
save_image(output, 'test_result.png')

'''------------------------Faking------------------------'''
for i in range(10):
	Alzheimer = model.generate_random_latent(stoc_dim).reshape([28,28])
	if (i == 0):
		fake_image = Alzheimer
	else:
		fake_image = torch.cat((fake_image, Alzheimer), dim = 1)	
save_image(fake_image, 'fake_result.png')

