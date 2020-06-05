
import torch
import numpy as np
from torch.autograd import Variable

def weight_variable(shape):
	return Variable(torch.randn(shape))

def bias_variable(shape):
	return Variable(torch.ones(shape) * 0.1)

def linear_then_tanh(input_x, W, b):
	return torch.tanh(torch.matmul(input_x, W) + b)

def linear(input_x, W, b):
	return torch.matmul(input_x, W) + b

def linear_then_exp(input_x, W, b):
	return torch.exp(torch.matmul(input_x, W) + b)

def linear_then_sigmoid(input_x, W, b):
	return torch.softmax((torch.matmul(input_x, W) + b), dim = 1)

def repeat(x, num, dim_):
    temp = x
    for i in range(num - 1):
        temp = torch.cat((temp,x), dim = dim_)
    return temp

def sample_from_gaussian(mean, var, batch_size):
	dimension = mean.shape[1]
	perturb = repeat(torch.randn([1, dimension]), batch_size, 0)
	return torch.mul(perturb, var) + mean


class importance_vae():
	def __init__(self, batch_size, stoc_dim, det_dim, K):
		self.stoc_dim = stoc_dim
		self.det_dim = det_dim
		self.K = K
		self.batch_size = batch_size
		self.p_param = []
		self.q_param = []
		self.K_samples = torch.tensor([])

	def construct_model(self, x):
		x_dim = x.shape[1]
		Q_1_variable = {'weight': weight_variable([x_dim, self.det_dim]), 'bias': bias_variable([self.batch_size, self.det_dim])}
		Q_2_variable = {'weight':weight_variable([self.det_dim, self.det_dim]), 'bias': bias_variable([self.batch_size, self.det_dim])}
		Q_3_variable = {'weight':weight_variable([self.det_dim, self.stoc_dim]),'bias': bias_variable([self.batch_size, self.stoc_dim])}
		self.q_param = [Q_1_variable, Q_2_variable, Q_3_variable]
		P_1_variable = {'weight':weight_variable([self.stoc_dim, self.det_dim]), 'bias': bias_variable([self.batch_size * self.K, self.det_dim])}
		P_2_variable = {'weight': weight_variable([self.det_dim, self.det_dim]), 'bias': bias_variable([self.batch_size * self.K, self.det_dim])}
		P_3_variable = {'weight': weight_variable([self.det_dim, 10]), 'bias': bias_variable([self.batch_size * self.K, 10])}
		self.p_param = [P_1_variable, P_2_variable, P_3_variable]
		params = 
		return self.q_param, self.p_param

	def encode(self, x):
		q_out_one = linear_then_tanh(x, self.q_param[0]['weight'], self.q_param[0]['bias'])
		q_out_two = linear_then_tanh(q_out_one, self.q_param[1]['weight'], self.q_param[1]['bias'])
		stoc_mean = linear(q_out_two, self.q_param[2]['weight'], self.q_param[2]['bias'])
		stoc_logvar = linear(q_out_two, self.q_param[2]['weight'], self.q_param[2]['bias'])
		return stoc_mean, stoc_logvar

	def get_K_samples(self, stoc_mean, stoc_logvar):
		stoc_var = torch.exp(stoc_logvar)
		for i in range(self.K):
			new_sample = sample_from_gaussian(stoc_mean, stoc_var, self.batch_size)
			self.K_samples = torch.cat((self.K_samples, new_sample), dim = 0)
		return self.K_samples
		
	def decode(self):
		p_out_one = linear_then_tanh(self.K_samples, self.p_param[0]['weight'], self.p_param[0]['bias'])
		p_out_two = linear_then_tanh(p_out_one, self.p_param[1]['weight'], self.p_param[1]['bias'])
		y_hat = linear_then_sigmoid(p_out_two, self.p_param[2]['weight'], self.p_param[2]['bias'])
		return y_hat


def IWAE_loss(K_samples, y_hat, batch_size, K, stoc_mean, stoc_logvar, y, stoc_dim):
	K_samples_tp = K_samples.reshape([K, batch_size, stoc_dim])
	y_hat_tp = y_hat.reshape([K, batch_size, 10])
	loss_per_batch = torch.ones(batch_size)
	for i in range(batch_size):
		h_i = K_samples_tp[:,i,:]
		mean_i = repeat(stoc_mean[i,:].reshape([1,-1]), K, dim_ = 0)
		var_i = torch.exp(repeat(stoc_logvar[i,:].reshape([1,-1]), K, dim_ = 0))
		q_h_given_x_i = torch.sum(1. / var_i * torch.exp(- 0.5 * torch.pow((h_i - mean_i) / var_i, 2)), dim = 1)
		y_hat_i = y_hat_tp[:,i,:]
		y_i = repeat(y[i,:].reshape([1,-1]), K, dim_ = 0)
		p_x_given_h_i = torch.exp(torch.sum(y_i * torch.log(y_hat_i) + (1. - y_i) * torch.log(1. - y_hat_i), dim = 1))
		loss_per_batch[i] = torch.log(torch.mean(p_x_given_h_i / q_h_given_x_i))
	return torch.mean(loss_per_batch)






