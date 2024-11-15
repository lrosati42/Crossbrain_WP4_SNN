import argparse

def get_args(t_tot = 500):

    args_dict = {
        't_tot': t_tot,     # interval time (ms). Used to build the result arrays
        'dt': 0.1,         # time step of simulation integration routine (ms)
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

    return argparse.Namespace(**args_dict)