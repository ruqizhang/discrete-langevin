import argparse
import rbm
import torch
import numpy as np
import samplers
import matplotlib.pyplot as plt
import os
import torchvision
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
import tensorflow_probability as tfp
import block_samplers
import time
import pickle
import itertools

def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_ess(chain, burn_in):
    c = chain
    l = c.shape[0]
    bi = int(burn_in * l)
    c = c[bi:]
    cv = tfp.mcmc.effective_sample_size(c).numpy()
    cv[np.isnan(cv)] = 1.
    return cv

def get_log_rmse(x,gt_mean):
    x = 2. * x - 1.
    x2 = ((x-gt_mean) ** 2).mean().sqrt()
    return x2.log().detach().cpu().numpy()

def tv(samples):
    gt_probs = np.load("{}/gt_prob_{}_{}.npy".format(args.save_dir,args.dim,args.bias))
    arrs, uniq_cnt = np.unique(samples, axis=0, return_counts=True)
    sample_probs = np.zeros_like(gt_probs)

    for i in range(arrs.shape[0]):
        sample_probs[i] = (uniq_cnt[i]*(1.)-1.)/samples.shape[0]
    l_dist =  np.abs((gt_probs - sample_probs)).sum()

def get_gt_mean(args,model):
    dim = args.dim**2
    A = model.J
    b = args.bias
    lst=torch.tensor(list(itertools.product([-1.0, 1.0], repeat=dim))).to(device)
    f = lambda x: torch.exp((x @ A * x).sum(-1)  + torch.sum(b*x,dim=-1))
    flst = f(lst)
    plst = flst/torch.sum(flst)
    gt_mean = torch.sum(lst*plst.unsqueeze(1).expand(-1,lst.size(1)),0)
    torch.save(gt_mean.cpu(),"{}/gt_mean_dim{}_sigma{}_bias{}.pt".format(args.save_dir,args.dim,args.sigma,args.bias))

    # gt_mean = torch.load("{}/gt_mean_dim{}_sigma{}_bias{}.pt".format(args.save_dir,args.dim,args.sigma,args.bias)).to(device)
    return gt_mean

def main(args):
    makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = rbm.LatticeIsingModel(args.dim, args.sigma, args.bias)
    model.to(device)
    gt_mean = get_gt_mean(args,model)

    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), 1, args.dim, args.dim),
                                                     p, normalize=False, nrow=int(x.size(0) ** .5))
    ess_samples = model.init_sample(args.n_samples).to(device)

    hops = {}
    ess = {}
    times = {}
    chains = {}
    means = {}
    rmses = {}
    x0 = model.init_dist.sample((args.n_test_samples,)).to(device)
    temps = [ 'hb-10-1', 'bg-1', 'gwg', 'dmala', 'dula']
    for temp in temps:
        if temp == 'dim-gibbs':
            sampler = samplers.PerDimGibbsSampler(model.data_dim)
        elif temp == "rand-gibbs":
            sampler = samplers.PerDimGibbsSampler(model.data_dim, rand=True)
        elif temp == "lb":
            sampler = samplers.PerDimLB(model.data_dim)
        elif "bg-" in temp:
            block_size = int(temp.split('-')[1])
            sampler = block_samplers.BlockGibbsSampler(model.data_dim, block_size)
        elif "hb-" in temp:
            block_size, hamming_dist = [int(v) for v in temp.split('-')[1:]]
            sampler = block_samplers.HammingBallSampler(model.data_dim, block_size, hamming_dist)
        elif temp == "gwg":
            sampler = samplers.DiffSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2.)
        elif "gwg-" in temp:
            n_hops = int(temp.split('-')[1])
            sampler = samplers.MultiDiffSampler(model.data_dim, 1,
                                                approx=True, temp=2., n_samples=n_hops)
        elif temp == "dmala":
            sampler = samplers.LangevinSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.4, mh=True)
        elif temp == "dula":
            sampler = samplers.LangevinSampler(model.data_dim, 1,
                                           fixed_proposal=False, approx=True, multi_hop=False, temp=2., step_size=0.2, mh=False)
        else:
            raise ValueError("Invalid sampler...")

        x = x0.clone().detach()
        times[temp] = []
        hops[temp] = []
        chain = []
        cur_time = 0.
        mean = torch.zeros_like(x)
        time_list = []
        rmses[temp] = []
        for i in range(args.n_steps):
            # do sampling and time it
            st = time.time()
            xhat = sampler.step(x.detach(), model).detach()
            cur_time += time.time() - st

            # compute hamming dist
            cur_hops = (x != xhat).float().sum(-1).mean().item()

            # update trajectory
            x = xhat

            mean = mean + x
            if i % args.subsample == 0:
                if args.ess_statistic == "dims":
                    chain.append(x.cpu().numpy()[0][None])
                else:
                    xc = x
                    h = (xc != ess_samples[0][None]).float().sum(-1)
                    chain.append(h.detach().cpu().numpy()[None])

            if i % args.viz_every == 0 and plot is not None:
                time_list.append(cur_time)
                rmse = get_log_rmse(mean / (i+1),gt_mean)
                rmses[temp].append(rmse)

            if i % args.print_every == 0:
                times[temp].append(cur_time)
                hops[temp].append(cur_hops)
        
        means[temp] = mean / args.n_steps
        chain = np.concatenate(chain, 0)
        chains[temp] = chain
        if not args.no_ess:
            ess[temp] = get_ess(chain, args.burn_in)
            print("ess = {} +/- {}".format(ess[temp].mean(), ess[temp].std()))
        np.save("{}/ising_sample_times_{}.npy".format(args.save_dir,temp),time_list)
        np.save("{}/ising_sample_logrmses_{}.npy".format(args.save_dir,temp),rmses)
    plt.clf()
    for temp in temps:
        plt.plot(rmses[temp], label="{}".format(temp))
    plt.legend()
    plt.savefig("{}/log_rmse.png".format(args.save_dir))

    if not args.no_ess:
        ess_temps = temps
        plt.clf()
        ess_list = [ess[temp] for temp in ess_temps]
        plt.boxplot(ess_list, labels=ess_temps, showfliers=False)
        plt.savefig("{}/ess.png".format(args.save_dir))
        plt.clf()
        plt.boxplot([ess[temp] / times[temp][-1] / (1. - args.burn_in) for temp in ess_temps], labels=ess_temps, showfliers=False)
        plt.savefig("{}/ess_per_sec.png".format(args.save_dir))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./figs/ising_sample")
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--n_samples', type=int, default=2)
    parser.add_argument('--n_test_samples', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234567)
    # model def
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--sigma', type=float, default=.1)
    parser.add_argument('--bias', type=float, default=0.2)
    # logging
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--viz_every', type=int, default=1000)
    # for rbm training
    parser.add_argument('--rbm_lr', type=float, default=.001)
    parser.add_argument('--cd', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--batch_size', type=int, default=100)
    # for ess
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--burn_in', type=float, default=.1)
    parser.add_argument('--ess_statistic', type=str, default="dims", choices=["hamming", "dims"])
    parser.add_argument('--no_ess', action="store_true")
    args = parser.parse_args()

    main(args)