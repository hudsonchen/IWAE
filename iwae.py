
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
	return torch.mul(perturb, var) + mean



class importance_vae(nn.Module):
	def __init__(self, feature_size, stoc_dim, det_dim):
		super(importance_vae, self).__init__()
		self.q_fc_1 = nn.Linear(feature_size, det_dim)

		self.q_fc_2 = nn.Linear(det_dim, det_dim)

		self.q_fc_mean = nn.Linear(det_dim, stoc_dim)
		
		self.q_fc_logvar = nn.Linear(det_dim, stoc_dim)

		self.p_fc_1 = nn.Linear(stoc_dim, det_dim)

		self.p_fc_2 = nn.Linear(det_dim, det_dim)

		self.p_fc_3 = nn.Linear(det_dim, feature_size)


	def encode(self, x):
		q_out_one = F.tanh(self.q_fc_1(x))
		q_out_two = F.tanh(self.q_fc_2(q_out_one))
		stoc_mean = self.q_fc_mean(q_out_two)
		stoc_logvar = self.q_fc_logvar(q_out_two)
		return stoc_mean, stoc_logvar

	def get_K_samples(self, stoc_mean, stoc_logvar, K, device):
		K_samples = torch.Tensor([]).to(device)
		stoc_var = torch.exp(stoc_logvar)
		for i in range(K):
			new_sample = sample_from_gaussian(stoc_mean, stoc_var, device)
			K_samples = torch.cat((K_samples, new_sample), dim = 0)
		return K_samples # （Batch_size * K）* stoc_dim
		
	def decode(self, K_samples):
		p_out_one = F.tanh(self.p_fc_1(K_samples))
		p_out_two = F.tanh(self.p_fc_2(p_out_one))
		X_hat = F.sigmoid(self.p_fc_3(p_out_two))
		return X_hat # (Batch_size * K) * stoc_dim

	def forward(self, x, K, device):
		stoc_mean, stoc_logvar = self.encode(x)
		K_samples = self.get_K_samples(stoc_mean, stoc_logvar, K, device)
		X_hat = self.decode(K_samples)
		return X_hat

	def generate_random_latent(self, stoc_dim):
		mean = torch.zeros([1, stoc_dim])
		var = torch.ones([1, stoc_dim])
		latent = sample_from_gaussian(mean, var)
		X_hat = self.decode(latent)
		return X_hat

	def IWAE_loss(self, x, X_hat, K_samples, batch_size, K, stoc_mean, stoc_logvar,  feature_size, stoc_dim, device):
		x_tp = repeat(x, K)
		K_samples_tp = K_samples.reshape([K, batch_size, stoc_dim])
		X_hat_tp = X_hat.reshape([K, batch_size, feature_size])
		mean_i = repeat(stoc_mean, K)
		var_i = torch.exp(repeat(stoc_logvar, K))
		log_q_h_given_xi = torch.sum(- torch.log(var_i) - 0.5 * torch.pow((K_samples_tp - mean_i) / var_i, 2), dim = 2)
		log_p_x_given_hi = torch.sum(x_tp * torch.log(X_hat_tp + eps) + (1. - x_tp) * torch.log(1. - X_hat_tp + eps), dim = 2)
		log_p_hi = torch.sum(-0.5 * (K_samples_tp **2), dim = 2)
		log_weight = log_p_x_given_hi + log_p_hi - log_q_h_given_xi
		weight_ = torch.softmax(log_weight, dim = 0)
		weight_ = Variable(weight_.detach(),requires_grad = False).to(device)
		ELBO = -torch.mean(torch.sum(weight_ * log_weight, dim = 1))
		return ELBO

	def VAE_loss(self, x, X_hat, K_samples, batch_size, K, stoc_mean, stoc_logvar,  feature_size, stoc_dim):
		x_tp = repeat(x, K)
		K_samples_tp = K_samples.reshape([K, batch_size, stoc_dim])
		X_hat_tp = X_hat.reshape([K, batch_size, feature_size])
		mean_i = repeat(stoc_mean, K)
		var_i = torch.exp(repeat(stoc_logvar, K))
		log_q_h_given_xi = torch.sum(- torch.log(var_i) - 0.5 * torch.pow((K_samples_tp - mean_i) / var_i, 2), dim = 2)
		log_p_x_given_hi = torch.sum(x_tp * torch.log(X_hat_tp + eps) + (1. - x_tp) * torch.log(1. - X_hat_tp + eps), dim = 2)
		log_p_hi = torch.sum(-0.5 * (K_samples_tp **2), dim = 2)
		log_weight = log_p_x_given_hi + log_p_hi - log_q_h_given_xi
		ELBO = -torch.mean(torch.mean(log_weight, dim = 1))
		return ELBO






