import torch
import numpy as np
from samplers import LangevinSamplerMultiDim
import argparse
import torch.nn as nn
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument('--sampler', type=str, default='dmala')
parser.add_argument('--n_steps', type=int, default=50000)
args = parser.parse_args()

class CategoricalDistribution(nn.Module):
    def __init__(self, data_dim, num_cls):
        super().__init__()
        self.data_dim = data_dim
        self.num_cls = num_cls

        prob_mat = torch.tensor(np.random.randn(self.data_dim, self.num_cls))
        self.prob_mat = prob_mat

    def forward(self, x):
        ### get log probability for state x
        ### shape of x: (batch_size, data_dim, num_cls); x must be a one-hot vector
        ### the log probability is (prob(x1) + prob(x3) + prob(x5) ... +prob(2k+1)) * (prob(x2) + prob(x4) + prob(2k))

        odd_prob = self.prob_mat[0::2, :]
        even_prob = self.prob_mat[1::2, :]

        odd_part = odd_prob[None, :, :] * x[:, 0::2, :]
        even_part = even_prob[None, :, :] * x[:, 1::2, :]

        odd_part = odd_part.view(odd_part.shape[0], -1).sum(dim=1, keepdim=False) 
        even_part = even_part.view(even_part.shape[0], -1).sum(dim=1, keepdim=False) 

        return odd_part - even_part

    def get_groundtruth_prob(self):
        zeros_mat = torch.zeros((1, self.data_dim, self.num_cls)) 

        classes = [i for i in range(self.num_cls)]
        probs = []
        for comb in product(classes, repeat=self.data_dim):
            cur_state = zeros_mat.detach().clone()
            cur_state[:, range(self.data_dim), comb] = 1.
            log_prob = self.forward(cur_state)
            #print(comb, log_prob.exp().item())
            print('Category:', comb)
            probs.append(log_prob.exp().item())
        probs = np.array(probs)
        print('Ground-Truth Distribution:\n', probs/probs.sum())

def x_to_index(x):
    index = torch.zeros_like(x)
    for i in range(DATA_DIM):
        index[:, i] = NUM_CLS**(DATA_DIM-i-1)
    #print(index)
    index = index * x
    index = index.sum(dim=1)
    #print(x)
    #print(index)
    return index.cpu().numpy()

DATA_DIM = 2 ### Number of dimensions
NUM_CLS = 3 ### Number of classes for each dimension
model = CategoricalDistribution(data_dim=2, num_cls=3)
model.get_groundtruth_prob()

if args.sampler == "dmala":
    sampler = LangevinSamplerMultiDim(model.data_dim, num_cls=model.num_cls, n_steps=1, temp=2., step_size=0.2, mh=True)
elif args.sampler == "dula":
    sampler = LangevinSamplerMultiDim(model.data_dim, num_cls=model.num_cls, n_steps=1, temp=2., step_size=0.1, mh=False)
else:
    assert False, 'Not implemented'

x = torch.tensor(np.random.randint(low=0, high=NUM_CLS, size=(1, DATA_DIM)))
#print(x.shape, x)

chain = []
frequency = np.zeros(NUM_CLS**DATA_DIM)

print('Start sampling with sampler:', args.sampler, ' Total iterations:', args.n_steps)
for i in range(args.n_steps):
    xhat = sampler.step(x.detach(), model).detach()
    x = xhat.long()
    index = x_to_index(x)
    for ind in index:
        frequency[ind] += 1

print('Empirical Frequency from Sampler:\n', frequency/frequency.sum())
