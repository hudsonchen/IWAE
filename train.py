# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:44:10 2020

@author: DELL
"""

import numpy as np
import iwae
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

def repeat(x, num, dim_):
    temp = x
    for i in range(num - 1):
        temp = torch.cat((temp,x), dim = dim_)
    return temp

batch_size = 20
batch_num = 1000
stoc_dim = 50
det_dim = 100
K = 2
feature_size = 28 * 28
epoch_num = 50
print_count = 0

train_image = np.load("train_data.npy")[:batch_size * batch_num,:].reshape([-1, batch_size, feature_size])
train_label = np.load("train_label.npy")
XX = train_image[batch_num - 1, batch_size - 1,:]

'''------------------------Construct model------------------------'''
model = iwae.importance_vae(feature_size, batch_size, stoc_dim, det_dim, K)


'''------------------------training------------------------'''
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

plt.imshow(XX.reshape([28,28]),cmap = 'gray')
plt.show()
	
eps = 1e-4
for j in range(epoch_num):
	for batch_idx in range(batch_num):
		train_batch_tensor = Variable(torch.tensor(train_image[batch_idx,:,:]))
		stoc_mean, stoc_logvar = model.encode(train_batch_tensor)
		#print(torch.mean(stoc_mean))
		K_samples = model.get_K_samples(stoc_mean, stoc_logvar, batch_size, K)
		X_hat = model.decode(K_samples)
	
		x = train_batch_tensor
		K_samples_tp = K_samples.reshape([K, batch_size, stoc_dim])
		X_hat_tp = X_hat.reshape([K, batch_size, feature_size])
		loss_per_batch = torch.ones(batch_size)
		for i in range(batch_size):
			h_i = K_samples_tp[:,i,:]
			mean_i = repeat(stoc_mean[i,:].reshape([1,-1]), K, dim_ = 0)
			var_i = torch.exp(repeat(stoc_logvar[i,:].reshape([1,-1]), K, dim_ = 0))
			log_q_h_given_xi = torch.sum(- torch.log(var_i) - 0.5 * torch.pow((h_i - mean_i) / var_i, 2), dim = 1)
			X_hat_i = X_hat_tp[:,i,:]
			x_i = repeat(x[i,:].reshape([1,-1]), K, dim_ = 0)
			log_p_x_given_hi = torch.sum(x_i * torch.log(X_hat_i + eps) + (1. - x_i) * torch.log(1. - X_hat_i + eps), dim = 1)
			log_p_hi = torch.sum(-0.5 * (h_i **2), dim = 1)
			log_tp = log_p_x_given_hi + log_p_hi - log_q_h_given_xi
			loss_per_batch[i] = torch.mean(log_tp)
			
		ELBO = -torch.mean(loss_per_batch)
		optimizer.zero_grad()
		ELBO.backward()
		optimizer.step()
	print(ELBO)
	plt.imshow(X_hat_i.detach().numpy()[1,:].reshape([28,28]),cmap = 'gray')
	plt.show()


