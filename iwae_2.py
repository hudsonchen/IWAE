import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-4

def repeat(x, num):
    temp = x
    for i in range(num - 1):
        temp = torch.cat((temp,x), dim = 0)
    return temp.reshape([-1, x.shape[0],x.shape[1]])

def sample_from_gaussian(mean, var, device):
	perturb = torch.randn_like(mean).to(device)
	return (torch.mul(perturb, var) + mean, perturb)

def get_K_samples(mean, logvar, K, device):
	K_samples = torch.Tensor([]).to(device)
	eps = torch.Tensor([]).to(device)
	var = torch.exp(logvar)
	for i in range(K):
		new_sample, perturb = sample_from_gaussian(mean, var, device)
		K_samples = torch.cat((K_samples, new_sample), dim = 0)
		eps = torch.cat((eps, perturb), dim = 0)
	eps_ = Variable(eps, requires_grad = False)
	return K_samples # （Batch_size * K）* stoc_dim

class importance_vae_2(nn.Module):
	def __init__(self, feature_size, batch_size, stoc_dim_1, stoc_dim_2, det_dim_1, det_dim_2):
		super(importance_vae_2, self).__init__()
		self.stoc_dim_1 = stoc_dim_1
		self.stoc_dim_2 = stoc_dim_2
		self.det_dim_1 = det_dim_1
		self.det_dim_2 = det_dim_2
		self.batch_size = batch_size
		self.feature_size = feature_size

		self.q_fc_1 = nn.Sequential(nn.Linear(feature_size, det_dim_1),
			nn.Tanh(),nn.Linear(det_dim_1, det_dim_1),nn.Tanh())

		self.q_fc_mean_h1 = nn.Linear(det_dim_1, stoc_dim_1)
		
		self.q_fc_logvar_h1 = nn.Linear(det_dim_1, stoc_dim_1)

		self.q_fc_2 = nn.Sequential(nn.Linear(stoc_dim_1, det_dim_2),
			nn.Tanh(),nn.Linear(det_dim_2, det_dim_2),nn.Tanh())

		self.q_fc_mean_h2 = nn.Linear(det_dim_2, stoc_dim_2)
		
		self.q_fc_logvar_h2 = nn.Linear(det_dim_2, stoc_dim_2)

		self.p_fc_1 = nn.Sequential(nn.Linear(stoc_dim_2, det_dim_2),
			nn.Tanh(),nn.Linear(det_dim_2, det_dim_2),nn.Tanh())

		self.p_fc_mean = nn.Linear(det_dim_2, stoc_dim_1)

		self.p_fc_logvar = nn.Linear(det_dim_2, stoc_dim_1)

		self.p_fc_2 = nn.Sequential(nn.Linear(stoc_dim_1, det_dim_1),
			nn.Tanh(),nn.Linear(det_dim_1, det_dim_1),nn.Tanh(),nn.Linear(det_dim_1, feature_size),nn.Sigmoid())

	def encode(self, x, K, device):
		q_out_one = self.q_fc_1(x)
		q_stoc_mean_h1 = self.q_fc_mean_h1(q_out_one)
		q_stoc_logvar_h1 = self.q_fc_logvar_h1(q_out_one)
		h1 = get_K_samples(q_stoc_mean_h1, q_stoc_logvar_h1, 1, device)
		q_out_two = self.q_fc_2(h1)
		q_stoc_mean_h2 = self.q_fc_mean_h2(q_out_two)
		q_stoc_logvar_h2 = self.q_fc_logvar_h2(q_out_two)
		h2 = get_K_samples(q_stoc_mean_h2, q_stoc_logvar_h2, K, device)
		return (h1, q_stoc_mean_h1, q_stoc_logvar_h1), (h2, q_stoc_mean_h2, q_stoc_logvar_h2)

	def decode(self, h1, h2):
		p_out_one = self.p_fc_1(h2)
		p_stoc_mean_h1 = self.p_fc_mean(p_out_one)
		p_stoc_logvar_h1 = self.p_fc_logvar(p_out_one)
		X_hat = self.p_fc_2(h1)
		return (p_stoc_mean_h1, p_stoc_logvar_h1, X_hat)

	def forward(self, x, K, device):
		(h1, q_stoc_mean_h1, q_stoc_logvar_h1), (h2, q_stoc_mean_h2, q_stoc_logvar_h2) = self.encode(x, K, device)
		(p_stoc_mean_h1, p_stoc_logvar_h1, X_hat) = self.decode(h1, h2)
		return X_hat

	def get_p_h(self, h1, h2, p_stoc_mean_h1, p_stoc_logvar_h1, K):
		h2 = h2.reshape([K, self.batch_size, self.stoc_dim_2])
		log_p_h2 = torch.sum(-0.5 * (h2 **2), dim = 2)
		p_stoc_mean_h1 = p_stoc_mean_h1.reshape([K, self.batch_size, self.stoc_dim_1])
		p_stoc_var_h1 = torch.exp(p_stoc_logvar_h1).reshape([K, self.batch_size, self.stoc_dim_1])
		h1 = repeat(h1, K)
		log_p_h1Gh2 = torch.sum(- torch.log(p_stoc_var_h1) - 0.5 * torch.pow((h1 - p_stoc_mean_h1) / p_stoc_var_h1, 2), dim = 2)
		#print('h1Gh2', log_p_h1Gh2)
		return (log_p_h1Gh2 + log_p_h2)

	def get_p_xGh(self, x, X_hat, K):
		x_tp = repeat(x, K)
		X_hat = repeat(X_hat, K)
		X_hat = X_hat.reshape([K, self.batch_size, self.feature_size])
		log_p_xGh = torch.sum(x_tp * torch.log(X_hat + eps) + (1. - x_tp) * torch.log(1. - X_hat + eps), dim = 2)
		return log_p_xGh

	def get_q_hGx(self, h1, q_stoc_mean_h1, q_stoc_logvar_h1, K):
		h1 = repeat(h1, K)
		q_stoc_mean_h1 = repeat(q_stoc_mean_h1, K)
		q_stoc_var_h1 = torch.exp(repeat(q_stoc_logvar_h1, K))
		log_q_hGx = torch.sum(- torch.log(q_stoc_var_h1) - 0.5 * torch.pow((h1 - q_stoc_mean_h1) / q_stoc_var_h1, 2), dim = 2)
		return log_q_hGx

	def get_q_h(self, h2, q_stoc_mean_h2, q_stoc_logvar_h2, K):
		h2 = h2.reshape([K, self.batch_size, self.stoc_dim_2])
		q_stoc_mean_h2 = repeat(q_stoc_mean_h2, K)
		q_stoc_var_h2 = torch.exp(repeat(q_stoc_logvar_h2, K))	
		log_q_h = torch.sum(- torch.log(q_stoc_var_h2) - 0.5 * torch.pow((h2 - q_stoc_mean_h2) / q_stoc_var_h2, 2), dim = 2)
		return log_q_h

	def IWAE_loss(self, log_p_h, log_p_xGh, log_q_hGx, log_q_h, device):
		log_weight = log_p_h + log_p_xGh - log_q_hGx - log_q_h
		weight_ = torch.softmax(log_weight, dim = 0)
		weight_ = Variable(weight_.detach(), requires_grad = False).to(device)
		ELBO = -torch.mean(torch.sum(weight_ * log_weight, dim = 1))
		return ELBO

	def VAE_loss(self, log_p_h, log_p_xGh, log_q_hGx, log_q_h):
		log_weight = log_p_h + log_p_xGh - log_q_hGx - log_q_h
		ELBO = -torch.mean(torch.mean(log_weight, dim = 1))
		return ELBO	
