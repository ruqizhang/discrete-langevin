# A Langevin-like Sampler for Discrete Distributions

This repository contains code for the paper
[A Langevin-like Sampler for Discrete Distributions](arxiv link), accepted in _International Conference on Machine Learning (ICML), 2022_.

```bibtex
@article{zhang2022langevinlike,
  title={A Langevin-like Sampler for Discrete Distributions},
  author={Zhang, Ruqi and Liu, Xingchao and Liu, Qiang},
  journal={International Conference on Machine Learning},
  year={2022}
}
```

# Introduction
We propose discrete Langevin proposal (DLP), a simple and scalable gradient-based
proposal for sampling complex high-dimensional discrete distributions. In contrast to Gibbs sampling-based methods, DLP is able to update all coordinates in parallel in a single step and the magnitude of changes is controlled by a stepsize. This allows a cheap and efficient exploration in the space of high-dimensional and strongly correlated variables. We prove the efficiency of DLP by showing that the asymptotic bias of the stationary distribution is zero for log-quadratic distributions, and is small for distributions that are close to being log-quadratic. With DLP, we develop several variants of sampling algorithms, including unadjusted, Metropolis-adjusted, stochastic and preconditioned versions. DLP outperforms many popular alternatives on a wide variety of tasks, including Ising models, restricted Boltzmann machines, deep energy-based models, binary neural networks and language generation.


# Dependencies
* [PyTorch 1.9.1](http://pytorch.org/) 
* [torchvision 0.10.1](https://github.com/pytorch/vision/)

# Usage
## Sampling From Ising Models
Please run
```
python ising_sample.py
```
## Sampling From Restricted Boltzmann Machines
Please run
```
python rbm_sample.py
```
## Learning Ising Models
Run ``bash generate_data.sh`` to generate the data, then learn the Ising model by running
```
python pcd.py --sampler=<SAMPLER>
```
* ```SAMPLER``` &mdash; Specify which sampler to use. \
                        ``dmala``: discrete Metropolis-adjusted Langevin algorithm; \
                        ``dula``: discrete unadjusted Langevin algorithm 

Use ``plt_pcd`` to plot the results of log RMSE with respect to the number of iterations and the runtime.

## Learning DEEP EBMS
The datasets can be found [here](https://github.com/jmtomczak/vae_vampprior/tree/master/datasets).

To learn the EBM, run ``bash ebm.sh`` and to evaluate the learned EBM using AIS, run ``ais.sh``.


## Binary Bayesian Neural Networks
See 
```
./BinaryBNN
```

# References
* This repo is built upon the [GWG repo](https://github.com/wgrathwohl/GWG_release) 
