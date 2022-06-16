import math
import os
import numpy as np
import random
import torch
from torch.optim import Adam, Adagrad, SGD
from torch.distributions import Normal
from torch.distributions.gamma import Gamma
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import adult_loader as ad
import compas_loader as cp
import blog_loader as bg
import news_loader as ns
import matplotlib.pyplot as plt

import argparse
from GWG_release import samplers

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

EPOCH = 1000+1
TEMP = 100.

parser = argparse.ArgumentParser()
parser.add_argument('--sampler', type=str, default='gibbs')
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=-1)
args = parser.parse_args()

setup_seed(args.seed)

log_dir = 'logs/%s/%s_%d_%d'%(args.dataset, args.sampler, args.batchsize, args.seed)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print(args.sampler)


class BayesianNN(nn.Module):
    def __init__(self, X_train, y_train, batch_size, num_particles, hidden_dim):
        super(BayesianNN, self).__init__()
        #self.lambda_prior = Gamma(torch.tensor(1., device=device), torch.tensor(1 / 0.1, device=device))
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.n_features = X_train.shape[1] 
        self.hidden_dim = hidden_dim

    def forward_data(self, inputs, theta):
        # Unpack theta
        w1 = theta[:, 0:self.n_features * self.hidden_dim].reshape(-1, self.n_features, self.hidden_dim)
        b1 = theta[:, self.n_features * self.hidden_dim:(self.n_features + 1) * self.hidden_dim].unsqueeze(1)
        w2 = theta[:, (self.n_features + 1) * self.hidden_dim:(self.n_features + 2) * self.hidden_dim].unsqueeze(2)
        b2 = theta[:, -1].reshape(-1, 1, 1)

        # num_particles times of forward
        inputs = inputs.unsqueeze(0).repeat(self.num_particles, 1, 1)
        inter = F.tanh(torch.bmm(inputs, w1) + b1)
        #print(inter.shape, w2.shape, b2.shape, self.hidden_dim, (self.n_features + 1) * self.hidden_dim)
        out_logit = torch.bmm(inter, w2) + b2
        out = out_logit.squeeze()
        out = torch.sigmoid(out)
        
        return out

    def forward(self, theta):
        theta = 2. * theta - 1.
        model_w = theta[:, :]
        # w_prior should be decided based on current lambda (not sure)
        w_prior = Normal(0., 1.)

        random_idx = random.sample([i for i in range(self.X_train.shape[0])], self.batch_size)
        X_batch = self.X_train[random_idx]
        y_batch = self.y_train[random_idx]

        outputs = self.forward_data(X_batch[:, :], theta)  # [num_particles, batch_size]
        y_batch_repeat = y_batch.unsqueeze(0).repeat(self.num_particles, 1)
        log_p_data = (outputs - y_batch_repeat).pow(2) 
        log_p_data = (-1.)*log_p_data.mean(dim=1)*TEMP

        #log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)
        #log_p = log_p0 + log_p_data  # (8) in paper
        log_p = log_p_data
        
        return log_p

def train_log(model, theta, X_test, y_test):
    with torch.no_grad():
        theta = 2. * theta - 1.
        model_w = theta[:, :]

        outputs = model.forward_data(X_test[:, :], theta)  # [num_particles, batch_size]
        y_batch_repeat = y_test.unsqueeze(0).repeat(model.num_particles, 1)
        log_p_data = (outputs - y_batch_repeat).pow(2) 
        log_p_data = (-1.)*log_p_data.mean(dim=1)

        #log_p0 = w_prior.log_prob(model_w.t()).sum(dim=0)
        #log_p = log_p0 + log_p_data / X_test.shape[0]  # (8) in paper
        log_p = log_p_data

        rmse = (outputs.mean(dim=0) - y_test).pow(2) 
        
        return log_p.mean().cpu().numpy(), rmse.mean().cpu().numpy()

def test_log(model, theta, X_test, y_test):
    with torch.no_grad():
        theta = 2. * theta - 1.
        model_w = theta[:, :]
        w_prior = Normal(0., 1.)

        outputs = model.forward_data(X_test[:, :], theta)  # [num_particles, batch_size]
        log_p_data = (outputs.mean(dim=0) - y_test).pow(2) 
        log_p_data = (-1.)*log_p_data.mean()

        log_p = log_p_data

        rmse = (outputs.mean(dim=0) - y_test).pow(2) 
        
        return log_p.mean().cpu().numpy(), np.sqrt(rmse.mean().cpu().numpy())



def main():
    device = 'cuda'
    if args.dataset == 'adult':
        X_train, y_train, X_test, y_test = ad.load_data(get_categorical_info=False)
    elif args.dataset == 'compas':
        X_train, y_train, X_test, y_test = cp.load_data(get_categorical_info=False)
    elif args.dataset == 'blog':
        X_train, y_train, X_test, y_test = bg.load_data(get_categorical_info=False)
    elif args.dataset == 'news':
        X_train, y_train, X_test, y_test = ns.load_data(get_categorical_info=False)
    else:
        print('Not Available')
        assert False

    n = X_train.shape[0]
    n = int(0.99*n)
    X_val = X_train[n:, :]
    y_val = y_train[n:]
    X_train = X_train[:n, :]
    y_train = y_train[:n]
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(np.max(X_train), np.min(X_train), np.mean(y_train), np.mean(y_test))

    feature_num = X_train.shape[1]
    X_train = torch.tensor(X_train).float().to(device)
    X_test = torch.tensor(X_test).float().to(device)
    X_val = torch.tensor(X_val).float().to(device)
    y_train = torch.tensor(y_train).float().to(device)
    y_test = torch.tensor(y_test).float().to(device)
    y_val = torch.tensor(y_val).float().to(device)

    X_train_mean, X_train_std = torch.mean(X_train[:, :], dim=0), torch.std(X_train[:, :], dim=0)
    X_train[:, :] = (X_train [:, :]- X_train_mean) / X_train_std
    X_test[:, :] = (X_test[:, :] - X_train_mean) / X_train_std
    
    if args.batchsize == -1:
        num_particles, batch_size, hidden_dim = 50, X_train.shape[0], 100 # 500 for others, 100 for blog
    else:
        num_particles, batch_size, hidden_dim = 50, args.batchsize, 100

    model = BayesianNN(X_train, y_train, batch_size, num_particles, hidden_dim)

    # Random initialization (based on expectation of gamma distribution)
    theta = torch.cat([torch.zeros([num_particles, (X_train.shape[1] +2) * hidden_dim + 1], device=device).normal_(0, math.sqrt(0.01))]) 
    theta = torch.bernoulli(torch.ones_like(theta)*0.5).to(device)
    print(theta.shape)
    dim = theta.shape[1]
    
    if args.sampler == 'gibbs':
        sampler = samplers.PerDimGibbsSampler(dim, rand=True)
    elif args.sampler == 'gwg':
        sampler = samplers.DiffSampler(dim, 1, fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
    elif args.sampler == 'langevin':
        sampler = samplers.LangevinSampler(dim, 1,fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.1, mh=False)
    elif args.sampler == 'langevin-mh':
        sampler = samplers.LangevinSampler(dim, 1,fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.1, mh=True)
    else:
        print('Not Available')
        assert False
    
    training_ll_cllt = []
    test_ll_cllt = []

    for epoch in range(EPOCH):
        theta_hat = sampler.step(theta.detach(), model).detach()
        theta.data = theta_hat.data.detach().clone()

        if epoch % 5 == 0:
            training_ll, training_rmse = train_log(model, theta, X_train, y_train)
            training_ll_cllt.append(training_ll)
            
            test_ll, test_rmse = test_log(model, theta, X_test, y_test)
            test_ll_cllt.append(test_rmse)
            if epoch % 100 == 0:
                print(epoch, 'Training LL:', training_ll, 'Test LL:', test_ll)
                print(epoch, 'Training RMSE:', training_rmse, 'Test RMSE:', test_rmse)

    np.save('%s/training_ll.npy'%(log_dir), np.array(training_ll_cllt))
    np.save('%s/test_rmse.npy'%(log_dir), np.array(test_ll_cllt))
    

if __name__ == '__main__':
    main()
