import torch
import torch.nn as nn
import torch.distributions as dists
import numpy as np



class LangevinSamplerMultiDim(nn.Module):
    def __init__(self, dim, num_cls=3, n_steps=10, multi_hop=False, temp=2., step_size=0.2, mh=True, device=None):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size  #rbm sampling: accpt prob is about 0.5 with lr = 0.2, update 16 dims per step (total 784 dims). ising sampling: accept prob 0.5 with lr=0.2
        # ising learning: accept prob=0.7 with lr=0.2
        # ebm: statistic mnist: accept prob=0.45 with lr=0.2
        
        self.mh = mh
        self.num_cls = num_cls ### number of classes in each dimension
    
        
    def get_grad(self, x, model):
        x = x.requires_grad_()
        gx = torch.autograd.grad(model(x).sum(), x)[0]
        return gx.detach()

    def to_one_hot(self, x):
        x_one_hot = torch.zeros((x.shape[0], self.dim, self.num_cls)).to(x.device)
        x_one_hot[:, range(self.dim), x[0, :]] = 1.

        return x_one_hot

    def step(self, x, model):
        '''
        input x : bs * dim, every dim contains a integer of 0 to (num_cls-1)
        '''
        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []        

        EPS = 1e-10
        for i in range(self.n_steps):
            x_cur_one_hot = self.to_one_hot(x_cur) 
            grad = self.get_grad(x_cur_one_hot, model) / self.temp
            
            ### we are going to create first term: bs * dim * num_cls, second term: bs * dim * num_cls
            grad_cur = grad[0:1, range(self.dim), x_cur[0, :]]
            first_term = grad.detach().clone() - grad_cur.unsqueeze(2).repeat(1, 1, self.num_cls) 
             
            second_term = torch.ones_like(first_term).to(x_cur.device) / self.step_size
            second_term[0, range(self.dim), x_cur[0, :]] = 0.

            cat_dist = torch.distributions.categorical.Categorical(logits=first_term-second_term)      
            x_delta = cat_dist.sample()

            if self.mh:
                lp_forward = torch.sum(cat_dist.log_prob(x_delta),dim=1)
                x_delta_one_hot = self.to_one_hot(x_delta) 
                grad_delta = self.get_grad(x_delta_one_hot, model) / self.temp
                
                grad_delta_cur = grad[0:1, range(self.dim), x_delta[0, :]]
                first_term_delta = grad_delta.detach().clone() - grad_delta_cur.unsqueeze(2).repeat(1, 1, self.num_cls) 
               
                second_term_delta = torch.ones_like(first_term_delta).to(x_delta.device) / self.step_size
                second_term_delta[0, range(self.dim), x_delta[0, :]] = 0.

                cat_dist_delta = torch.distributions.categorical.Categorical(logits=first_term_delta - second_term_delta)      
                lp_reverse = torch.sum(cat_dist_delta.log_prob(x_cur),dim=1)
                
                m_term = (model(x_delta_one_hot).squeeze() - model(x_cur_one_hot).squeeze())
                la = m_term + lp_reverse - lp_forward
                a = (la.exp() > torch.rand_like(la)).float()
                a_s.append(a.mean().item())
                x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            else:
                x_cur = x_delta

        return x_cur

