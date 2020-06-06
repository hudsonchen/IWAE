
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn

eps = 1e-4
def weight_parameter(shape):
	std = torch.ones(shape) * 0.1
	return nn.Parameter(torch.normal(0.0, std))

def bias_parameter(shape):
	return nn.Parameter(torch.zeros(shape))

def linear_then_tanh(input_x, W, b):
	return torch.tanh(torch.matmul(input_x, W) + b)

def linear(input_x, W, b):
	return torch.matmul(input_x, W) + b

def linear_then_exp(input_x, W, b):
	return torch.exp(torch.matmul(input_x, W) + b)

def linear_then_sigmoid(input_x, W, b):
	return torch.sigmoid((torch.matmul(input_x, W) + b))

def repeat(x, num, dim_):
    temp = x
    for i in range(num - 1):
        temp = torch.cat((temp,x), dim = dim_)
    return temp

def sample_from_gaussian(mean, var, batch_size):
	dimension = mean.shape[1]
	perturb = repeat(torch.randn([1, dimension]), batch_size, 0)
	return torch.mul(perturb, var) + mean


class importance_vae(nn.Module):
	def __init__(self, feature_size, batch_size, stoc_dim, det_dim, K):
		super(importance_vae, self).__init__()
		self.Q_1_weight = weight_parameter([feature_size, det_dim])
		self.Q_1_bias = bias_parameter([batch_size, det_dim])

		self.Q_2_weight = weight_parameter([det_dim, det_dim])
		self.Q_2_bias = bias_parameter([batch_size, det_dim])

		self.Q_3_weight = weight_parameter([det_dim, stoc_dim])
		self.Q_3_bias = bias_parameter([batch_size, stoc_dim])
		
		self.Q_4_weight = weight_parameter([det_dim, stoc_dim])
		self.Q_4_bias = bias_parameter([batch_size, stoc_dim])

		self.P_1_weight = weight_parameter([stoc_dim, det_dim])
		self.P_1_bias = bias_parameter([batch_size * K, det_dim])

		self.P_2_weight = weight_parameter([det_dim, det_dim])
		self.P_2_bias = bias_parameter([batch_size * K, det_dim])

		self.P_3_weight = weight_parameter([det_dim, feature_size])
		self.P_3_bias = bias_parameter([batch_size * K, feature_size])


	def encode(self, x):
		q_out_one = linear_then_tanh(x, self.Q_1_weight, self.Q_1_bias)
		q_out_two = linear_then_tanh(q_out_one, self.Q_2_weight, self.Q_2_bias)
		stoc_mean = linear(q_out_two, self.Q_3_weight, self.Q_3_bias)
		stoc_logvar = linear(q_out_two, self.Q_4_weight, self.Q_4_bias) / 10
		return stoc_mean, stoc_logvar

	def get_K_samples(self, stoc_mean, stoc_logvar, batch_size, K):
		K_samples = torch.tensor([])
		stoc_var = torch.exp(stoc_logvar)
		for i in range(K):
			new_sample = sample_from_gaussian(stoc_mean, stoc_var, batch_size)
			K_samples = torch.cat((K_samples, new_sample), dim = 0)
		return K_samples
		
	def decode(self, K_samples):
		p_out_one = linear_then_tanh(K_samples, self.P_1_weight, self.P_1_bias)
		p_out_two = linear_then_tanh(p_out_one,  self.P_2_weight, self.P_2_bias)
		X_hat = linear_then_sigmoid(p_out_two,  self.P_3_weight, self.P_3_bias)
		return X_hat


def IWAE_loss(x, X_hat, K_samples, batch_size, K, stoc_mean, stoc_logvar,  feature_size, stoc_dim):
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
		log_p_x_given_hi = torch.mean(x_i * torch.log(X_hat_i + eps) + (1. - x_i) * torch.log(1. - X_hat_i + eps), dim = 1)
		log_p_hi = torch.mean(-0.5 * (h_i **2), dim = 1)
		log_tp = log_p_x_given_hi + log_p_hi - log_q_h_given_xi
		loss_per_batch[i] = torch.log(torch.mean(torch.exp(log_tp)))
	return torch.mean(loss_per_batch)






