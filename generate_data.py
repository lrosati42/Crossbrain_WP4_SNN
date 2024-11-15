import numpy as np
from epileptor import classes
from argparse import ArgumentParser, Namespace
import time
from datetime import timedelta
from pathlib import Path

# ----- TIME REFERENCE -----
start = time.time()

parser = ArgumentParser()
parser.add_argument("-len", "--length", dest="sim_len", type=int,
                    help="duration of the simulation (s)")
parser.add_argument("-s", "--seed", dest="seed", default=42, type=int,
                    help="seed of random number generator")
args = parser.parse_args()

np.random.seed(args.seed)

def get_args(t_tot = 3600*1e3):

    epileptor_params = {
        't_tot': t_tot,     # interval time (ms). Used to build the result arrays
        'dt': 0.1,          # time step of simulation integration routine (ms)
        'CpES1': 0.8,       # Coupling % of electrical synapses (i.e. gap junctions) [0-1] within population 1 neurons
        'CpES2': 0.8,       # Coupling % of electrical synapses (i.e. gap junctions) [0-1] within population 2 neurons
        'CpCS': 1.,         # Coupling % of chemical synapses [0-1]
        'm': 0.8,           # m parameter in the epileptor equations
        'x0': -2.,          # x0 in the epileptor equations
        'r': 0.0001,        # r parameter in population 1 neurons equations
        's': 8,             # s parameter in population 1 neurons equations, reflecting electrical impact rate on slow variable
        'nbn1': 40,         # number of neurons in population 1
        'nbn2': 40,         # number of neurons in population 2
        'g_x1x1': 0.2,      # collateral synaptic (maximum) conductance between neurons from population 1
        'g_x2x2': 0.2,      # collateral synaptic (maximum) conductance between neurons from population 2
        'g_x1x2': 0.2,      # synaptic (maximum) conductance between neurons from population 1 to population 2
        'g_x2x1': 0.2,      # fast synaptic (maximum) conductance between neurons from population 2 to population
        'g_x2x1_slow': 0,   # slow synaptic (maximum) conductance between neurons from population 2 to population 1
        'g_x2x2_slow': 0,   # slow synaptic (maximum) conductance between neurons from population 2 to population 1
        'I2': 0.8,          # baseline input current in population 2 neurons
        'I1': 3.1,          # baseline input current in population 1 neurons
        'c2': 0.3,          # scaling factor of z injection in pop2 neurons
        'noise1': 0.5,      # noise amplitude that is introduced in population 1 neurons at each time step
        'noiseRatio': 5,    # noise factor for population 1 as compared to population 2
        'noise2': 0.3,      # noise amplitude that is introduced in population 2 neurons at each time step
        'noise3': 0.,       # noise amplitude that is introduced in slow variable at each time step
        'fs': 500,          # sampling rate (Hz)
        'n': 40,            # number of samples to get data at the desired sampling frequency = 1/(dt*n)
    }
    epileptor_params["n"] = ((1/epileptor_params["dt"]) / epileptor_params["fs"]) * 1000
    return Namespace(**epileptor_params)

# epileptor parameters
t_tot = args.sim_len*1e3    # ms
steps = t_tot*2     # @ 500 Hz
epi_args = get_args(t_tot=t_tot)

duration = 500      # ms
smpl_freq = 500     # Hz
timepoints = int(duration * smpl_freq / 1e3)
dt = (1 / smpl_freq) * 1e3
sim_steps = int(duration/epi_args.dt)

time = np.arange(0, t_tot, duration)
time_sim = np.arange(0, t_tot, epi_args.dt)

exp_id = 0
inpath = f'/srv/nfs-data/picard/luigi/epileptor/parameters/{exp_id}'
x0 = np.load(f'{inpath}/x0_real.npy')[:time.size]
CpES = np.load(f'{inpath}/cp_real.npy')[:time.size]

x0_sim = np.interp(time_sim, time, x0)
CpES_sim = np.interp(time_sim, time, CpES)

# instantiate epileptor model
epinet = classes.SeizureNetwork(epi_args)
epinet.initialize_networks()

##### SET INPUT SPIKETIMES #####
x1_i, x2_i, evs_i = epinet.advance_simulation(t_stop=t_tot, x0_variable=x0_sim, CpES_variable=CpES_sim)
x1_i = x1_i.mean(axis=0)
x2_i = x2_i.mean(axis=0)
input_signal = 0.8*x1_i + 0.2*x2_i
input_signal_norm = (input_signal - input_signal.min()) / (input_signal.max() - input_signal.min())

# save results
expath = f'data/synthetic/epileptor/full/{exp_id}'
Path(expath).mkdir(parents=True, exist_ok=True)
np.save(f'{expath}/mean_nonorm.npy', input_signal)
np.save(f'{expath}/mean.npy', input_signal_norm)
np.save(f'{expath}/st_ei.npy', evs_i)

# ----- TIME REFERENCE -----
end = time.time()
print(f'Code ended in {str(timedelta(seconds=(end-start)))}')
