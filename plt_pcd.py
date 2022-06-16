import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set_style("whitegrid")
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np

## ising learning
gibbs = np.load('./figs/ising_learn/rmse_gibbs_0.25_100.npy')
gwg = np.load('./figs/ising_learn/rmse_gwg_0.25_100.npy')
dmala = np.load('./figs/ising_learn/rmse_dmala_0.25_100.npy')
dula = np.load('./figs/ising_learn/rmse_dula_0.25_100.npy')
gibbs = [np.log(t) for t in gibbs]
gwg = [np.log(t) for t in gwg]
dmala = [np.log(t) for t in dmala]
dula = [np.log(t) for t in dula]

x= range(len(gibbs))
x=[t*1000 for t in x]
plt.plot(x,gibbs,lw=2,label='Gibbs-1')
plt.plot(x,gwg,lw=2,label='GWG-1')
plt.plot(x,dmala,lw=2,label='DMALA')
plt.plot(x,dula,lw=2,label='DULA')
plt.xlabel('Iters ',fontsize=17)
# plt.xscale('log')
plt.legend(fontsize=18)
plt.ylabel('log RMSE',fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim(left=0)
plt.savefig('figs/ising_learn/logrmse_.25.pdf')
plt.close()

#time
plt.clf()
gibbst = np.load('./figs/ising_learn/times_gibbs_0.25_100.npy')
gwgt = np.load('./figs/ising_learn/times_gwg_0.25_100.npy')
dmalat = np.load('./figs/ising_learn/times_dmala_0.25_100.npy')
dulat = np.load('./figs/ising_learn/times_dula_0.25_100.npy')
plt.plot(gibbst,gibbs,lw=2,label='Gibbs-1')
plt.plot(gwgt,gwg,lw=2,label='GWG-1')
plt.plot(dmalat,dmala,lw=2,label='DMALA')
plt.plot(dulat,dula,lw=2,label='DULA')
plt.xlabel('Runtime (s)',fontsize=17)
plt.legend(fontsize=18)
plt.ylabel('log RMSE',fontsize=17)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.xlim(left=0,right=1500)
plt.savefig('figs/ising_learn/time_logrmse_.25.pdf')
plt.close()