from argparse import ArgumentParser
from pathlib import Path
import csv
from model import LSM_epi_nosp as LSM
from epileptor_params import get_args
from epileptor import classes
from data_loader import epileptor
# from par_space_dynamics import Parameter_Walker as Par_dyn
from coding import population_coding, FR_decoding
from train import Suptrain_data as train
from test import Suptest_data as test
import time
from datetime import timedelta
import numpy as np
from _static.utils import set_weights
import pyNN.neuron as pynn
pynn.setup(timestep=0.1)

parser = ArgumentParser()
parser.add_argument("-N", "--number", dest="n_rec", type=int,
                    help="number of reservoir neurons")
parser.add_argument("-len", "--length", dest="sim_len", type=int,
                    help="duration of the simulation (s)")
parser.add_argument("-exp", "--experiment", dest="n_exp", type=int,
                    help="marix realization")
parser.add_argument("-mod", "--modality", default='rand', dest="mod", type=str,
                    help="Pure random or Orthonormal connectivity matrix. Use 'ortho', 'rand' or 'dale'")
parser.add_argument("-spar", "--sparsity", default='0', dest="spar", type=str,
                    help="Sparsity level in the connectivity matrix. Explored levels: 0, 20, 50, 80")
parser.add_argument("-s", "--seed", dest="seed", default=42, type=int,
                    help="seed of random number generator")
args = parser.parse_args()

# simulation parameters
params = {
    "n_rec": int(args.n_rec),       # number of recurrent neurons
    "t_exp": 1,
    "n_in": 21,                     # number of input neurons
    # "n_smpl": 20,                   # number of sampled input neuron
    # "n_subs": 4,
    "n_out": 2,                     # number of readout neurons (classes)
    "mu_in_rec": 0.5,               # weights mean from input to recurrent network 0
    "sigma_in_rec": 0.6,            # weights std from input to recurrent network  5
    # "mu_smpl_rec": 1.,              # weights mean from input to recurrent network  1
    # "sigma_smpl_rec": 0.5,          # weights std from input to recurrent network   0.5    
    "mu_rec_rec": 0.,               # weights mean of recurrent connections
    "sigma_rec_rec": 0.3,           # weights std of recurrent connections, goes as 1/(n_main)**(1/2)
    "lambda": 70.,
    "mu_rec_out": 0,                # weights mean from recurrent to readout network
    "sigma_rec_out": 0.1,           # weights std from recurrent to readout network
    "total_duration": args.sim_len * 1e3,   # total simulation duration (ms)
    "duration": 0.5 * 1e3,          # single simulation step duration (ms)
    "sample_rate": 500,             # Hz
    "s_pop": 1.,
    "tau_mem": 8,                   # membrane time constant (s)
    "tau_syn": 2,                   # synaptic time constant (s)
    "tau_out": 5,                   # readout time constant (s)
    "n_epochs": 10,                 # number of epochs in the training loop
    "eta": 0.003,                   # 1e-3
    "d_max": 20                     # maximum value of the duv=0.1 liquid state (empiric threshold)
    }

# time reference
start = time.time()

np.random.seed(args.seed)

sim_len = int(args.sim_len) # s
n_exp = int(args.n_exp)

# load data
datapath = parpath = 'data/synthetic/epileptor/full/'
# instantiate data loader
loader = epileptor(parpath)
x0, CpES = loader.load('x0_real.npy', 'cp_real.npy')
# mean activity
mean = np.load(datapath + 'mean.npy')

## split into train and test set
n_slices = int(sim_len / params["duration"] * 1e3)
x0 = x0[:n_slices]
CpES = CpES[:n_slices]
mean = mean[:n_slices*250]

perc_test = 0.2
n_test_epochs = int(perc_test * n_slices)
n_epochs = n_slices - n_test_epochs
# idxtrain = np.random.choice(x0.size, size=n_epochs, replace=False)
idxtrain = np.arange(n_epochs)
train_x0 = x0[idxtrain]
train_CpES = CpES[idxtrain]
mean_train = mean.reshape(250, int(mean.size/250))[:,idxtrain]
mean_train = mean_train.flatten()
idxtest = np.setdiff1d(np.arange(x0.size), idxtrain)
n_test_epochs = idxtest.size
test_x0 = x0[idxtest]
test_CpES = CpES[idxtest]
mean_test = mean.reshape(250, int(mean.size/250))[:,idxtest]
mean_test = mean_test.flatten()

# epileptor parameters
epi_args = get_args()
epi_args.t_tot = sim_len * 1000     # ms
epi_args.dt = 0.1                   # ms
epi_args.fs = 500                   # Hz
epi_args.n = ((1/epi_args.dt) / epi_args.fs) * 1000

# instantiate epileptor model
epinet = classes.SeizureNetwork(epi_args)
epinet.initialize_networks()

# instantiate epileptor decoder model
decoder = classes.SeizureNetwork(epi_args)
decoder.initialize_networks()

# instantiate LSM model
net = LSM(n_in = params["n_in"], n_smpl = None, n_subs = None, n_res = params["n_rec"], n_out = params["n_out"])
net.create()
net.connect()
net.initialize_weights()

n_in = params["n_in"]
n_rec = params["n_rec"]
n_out = params["n_out"]

# instantiate parameters dynamics
# pardyn = Par_dyn(ds=0.015, x0_0 = 0.1, CpES_0 = 0.1)

# LOAD connectivity matrices
basematrixpath = f'data/synthetic/epileptor/matrices/{n_rec}'
matrixpath = f'{basematrixpath}/{args.mod}/spar{args.spar}'
Jin = np.load(f'{basematrixpath}/Jin.npy')
Jrec = np.load(f'{matrixpath}/cc{n_exp}.npy')
Jout = np.zeros((n_rec, n_out))

# set weights
set_weights(Jin, net.in_rec_inh, net.in_rec_exc)
set_weights(Jrec, net.proj_J_inh, net.proj_J_exc)
set_weights(Jout, net.rec_out_inh, net.rec_out_exc)

# create export folder for results
csvpath = "results/epileptor/full/noplast"
Path(csvpath).mkdir(parents=True, exist_ok=True)
expath = f"{csvpath}/{n_rec}/{args.mod}/spar{args.spar}"
Path(expath).mkdir(parents=True, exist_ok=True)

coding = population_coding(n_in=net.n_in, s_pop=params["s_pop"])
decoding = FR_decoding(fmin = 0, fmax = 100)

##### LSM TRAINING #####
# instantiate training class
LSMtrainer = train()
# TRAINING LOOP
print(f"Training the n{n_exp} LSM with size {n_rec}")
LSMtrainer.training_loop(params, net, coding, decoding, epinet, mean_train, train_x0, train_CpES, decoder, n_epochs)

##### LSM TEST #####
# instantiate test class
LSMtester = test()
# TEST LOOP
LSMtester.test_loop(params, net, coding, decoding, epinet, mean_test, test_x0, test_CpES, decoder, n_test_epochs)

# get final matrix
Jout_e = net.rec_out_exc.get("weight", format="array")
Jout_i = net.rec_out_inh.get("weight", format="array")
Jout0 = Jout_e + Jout_i
np.save(f'{expath}/Jout_{n_exp}.npy', Jout0)
np.save(f'{expath}/x0_o_{n_exp}.npy', LSMtester.out_x0)
np.save(f'{expath}/cp_o_{n_exp}.npy', LSMtester.out_CpES)
np.save(f'{expath}/idxtest_{n_exp}.npy', idxtest)

# save results
header_labels = ["N", "T", "mod", "spar", "LSM_loss", "LSM_loss_s"]
final_dict = {
    "N": n_rec,
    "T": sim_len,
    "mod": args.mod,
    "spar": args.spar,
    "LSM_loss": LSMtester.loss.mean(),
    "LSM_loss_s": LSMtester.loss.std(ddof=1)
    }
with open(f'{csvpath}/final.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = header_labels)
    writer.writerow(final_dict)

# time reference
end = time.time()

pynn.end()

print(f'Code ended in {str(timedelta(seconds=(end-start)))}')
