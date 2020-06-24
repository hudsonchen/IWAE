
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
eps = 1e-4

def generate_data(D, N):
	mu_true = torch.randn([D,1])
	x = torch.zeros([N,D])
	for i in range(N):
		x[i,:] = torch.normal(mu_true, 1).reshape([1,D])
	return x, mu_true

def repeat(x, num):
    temp = x
    for i in range(num - 1):
        temp = torch.cat((temp,x), dim = 0)
    return temp.reshape([-1, x.shape[0]])

def add_noise(x):
	x_ones = torch.ones_like(x)
	noise = torch.normal(0, x_ones * 0.01)
	return x + noise

class SNR_iwae(nn.Module):
	def __init__(self, N, D, batch):
		super(SNR_iwae, self).__init__()
		self.N = N
		self.D = D
		self.batch = batch

	def get_loss(self, x, A, b, K, theta, flag):
		z = torch.mm(x, A) + b.unsqueeze(0).expand(x.shape[0], -1)
		log_xGz = -0.5 * torch.sum((x - z) ** 2, dim = 1)
		log_z = torch.zeros_like(log_xGz)
		for i in range(self.N):
			log_z[i] = -0.5 * torch.sum((z[i,:] - theta) ** 2)
		log_weight = log_xGz + log_z
		if (flag == 'IWAE'):
			log_weight = log_weight[:self.batch * K].reshape([self.batch, K])
			weight = torch.softmax(log_weight, dim = 1)
			weight_ = Variable(weight.detach(), requires_grad = False)
			ELBO = -torch.mean(torch.sum(weight_ * log_weight, dim = 1))
		elif (flag == 'VAE'):
			log_weight = log_weight[:self.batch * K].reshape([self.batch, K])
			ELBO = -torch.mean(torch.mean(log_weight))
		else:
			print('Nice try!')
		return ELBO

'''------------------------Hyper-parameter------------------------'''
K_all = [1, 3, 10, 70, 200]
N = 1024
D = 20
loop = 1000
eps = 1e-4
batch = 1
flag = 'IWAE'

'''------------------------Loading training data------------------------'''
x_train, mu_true = generate_data(D, N)
A_ = torch.eye(D) * 0.5
b_ = torch.mean(x_train, dim = 0) * 0.5
theta_ = torch.mean(x_train, dim = 0)
grad_A = torch.zeros(loop, 5)
grad_theta = torch.zeros(loop, 5)
x_train_tensor = Variable(torch.Tensor(x_train))
'''------------------------Construct model------------------------'''

for i in range(loop):
	model = SNR_iwae(N, D, batch)
	A = Variable(add_noise(A_), requires_grad = True)
	b = Variable(add_noise(b_), requires_grad = True)
	theta = Variable(add_noise(theta_), requires_grad = True)
	for j in range(len(K_all)):
		Loss = model.get_loss(x_train_tensor, A, b, K_all[j], theta, flag)
		Loss.backward()
		grad_A[i, j] = A.grad[1,1]
		grad_theta[i, j] = theta.grad[1]
		A.grad.data.zero_()
		b.grad.data.zero_()
		theta.grad.data.zero_()
	print(i)
	
plt.figure()
plt.hist(grad_A[:,0], bins = 30, color='fuchsia',alpha = 0.4)
plt.hist(grad_A[:,1], bins = 30, color='forestgreen',alpha = 0.4)
plt.hist(grad_A[:,2], bins = 30, color='indigo',alpha = 0.4)
plt.hist(grad_A[:,3], bins = 30, color='yellow',alpha = 0.4)
plt.hist(grad_A[:,4], bins = 30, color='skyblue',alpha = 0.4)
plt.xlabel('$\nabla(b)$')
plt.ylabel('$b$-density')
plt.savefig('VAE gradient for inference network.jpg')
plt.figure()
plt.hist(grad_theta[:,0], bins = 30, color='fuchsia',alpha = 0.4)
plt.hist(grad_theta[:,1], bins = 30, color='forestgreen',alpha = 0.4)
plt.hist(grad_theta[:,2], bins = 30, color='indigo',alpha = 0.4)
plt.hist(grad_theta[:,3], bins = 30, color='yellow',alpha = 0.4)
plt.hist(grad_theta[:,4], bins = 30, color='skyblue',alpha = 0.4)
plt.xlabel('$\nabla(\theta)$')
plt.ylabel('$\theta$-density')
plt.savefig('VAE gradient for generative network.jpg')
plt.show()
