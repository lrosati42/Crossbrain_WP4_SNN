'''
Created on 20 juil. 2012

@author: Squirel
'''

import numpy as np
from random import uniform

from epileptor import pop1n
from epileptor import pop2n

# import argparse
# import matplotlib.pyplot as plt

def find_threshold_indices(array, threshold):
    above_threshold_indices = []
    above_threshold = False
    for i, value in enumerate(array):
        if value > threshold:
            if not above_threshold:
                above_threshold_indices.append(i)
                above_threshold = True
        else:
            above_threshold = False
    return above_threshold_indices

def filter_indices(indices, threshold):
    filtered_indices = []
    if not indices:  # If the input list is empty, return empty list
        return filtered_indices
    
    first_index = indices[0]  # First kept index
    filtered_indices.append(first_index)
    
    for index in indices[1:]:
        if index - first_index > threshold:
            filtered_indices.append(index)
            first_index = index  # Update the first index to the current index
    
    return filtered_indices

class SeizureNetwork:
    def __init__(self, args):
        self.parameters = Parameters(args)
        self.initialize_arrays()

    def initialize_arrays(self):
        nbn1 = self.parameters.nbn1
        nbn2 = self.parameters.nbn2
        t_tot = self.parameters.t_tot
        dt = self.parameters.dt
        n = self.parameters.n

        self.x1_plot = np.zeros((nbn1, int(t_tot / dt / n)))
        self.x2_plot = np.zeros((nbn2, int(t_tot / dt / n)))
        self.y1_plot = np.zeros((nbn1, int(t_tot / dt / n)))
        self.y2_plot = np.zeros((nbn2, int(t_tot / dt / n)))
        self.x1bar_plot = np.zeros(int(t_tot / dt / n))
        self.x2bar_plot = np.zeros(int(t_tot / dt / n))
        self.y1bar_plot = np.zeros(int(t_tot / dt / n))
        self.y2bar_plot = np.zeros(int(t_tot / dt / n))
        self.z_plot = np.zeros((nbn1, int(t_tot / dt / n)))
        self.zbar_plot = np.zeros(int(t_tot / dt / n))
        self.KOP_plot = np.zeros((2, int(t_tot / dt / n)))
        self.evs = np.empty(shape=(1,2))

        # arrays to store means between samples
        self.x1_nsamples = np.zeros((nbn1, n))
        self.x2_nsamples = np.zeros((nbn2, n))
        self.y1_nsamples = np.zeros((nbn1, n))
        self.y2_nsamples = np.zeros((nbn2, n))
        self.z_nsamples = np.zeros((nbn1, n))
        self.x1_nsamples_i = np.zeros((nbn1))
        self.x2_nsamples_i = np.zeros((nbn2))
        self.y1_nsamples_i = np.zeros((nbn1))
        self.y2_nsamples_i = np.zeros((nbn2))
        self.z_nsamples_i = np.zeros((nbn1))
        self.x1bar_nsamples = np.zeros(n)
        self.x2bar_nsamples = np.zeros(n)
        self.y1bar_nsamples = np.zeros(n)
        self.y2bar_nsamples = np.zeros(n)
        self.zbar_nsamples = np.zeros(n)
        self.KOP_nsamples = np.zeros((2, n))


    def initialize_networks(self):
        nbn1 = self.parameters.nbn1
        nbn2 = self.parameters.nbn2
        CpES1 = self.parameters.CpES1
        CpES2 = self.parameters.CpES2
        CpCS = self.parameters.CpCS
        g_x1x1 = self.parameters.g_x1x1
        g_x2x2 = self.parameters.g_x2x2
        g_x1x2 = self.parameters.g_x1x2
        g_x2x1 = self.parameters.g_x2x1
        g_x2x1_slow = self.parameters.g_x2x1_slow
        g_x2x2_slow = self.parameters.g_x2x2_slow
        I1 = self.parameters.I1
        I2 = self.parameters.I2
        c2 = self.parameters.c2
        m = self.parameters.m
        x0 = self.parameters.x0
        r = self.parameters.r
        s = self.parameters.s
        noise1 = self.parameters.noise1
        noise2 = self.parameters.noise2
        noise3 = self.parameters.noise3

        # set random initial conditions
        x1_init = [uniform(-1., 1.5) for _ in range(nbn1)]
        y1_init = [uniform(-5., 0.) for _ in range(nbn1)]
        z_init = [uniform(3., 3.) for _ in range(nbn1)]
        x2_init = [uniform(-1.25, 1.) for _ in range(nbn2)]
        y2_init = [uniform(0., 1.) for _ in range(nbn2)]

        self.pop1 = [pop1n.pop1n(m=m, x0=x0, CpES=CpES1, CpCS=CpCS, g_x1x1=g_x1x1, g_x2x1=g_x2x1, g_x2x1_slow=g_x2x1_slow,
                                 I1=I1, r=r, s=s, noise=noise1, noise3=noise3) for _ in range(nbn1)]
        for i in range(nbn1):
            self.pop1[i].x1 = x1_init[i]
            self.pop1[i].y1 = y1_init[i]
            self.pop1[i].z = z_init[i]

        self.pop2 = [pop2n.pop2n(CpES=CpES2, CpCS=CpCS, g_x2x2=g_x2x2, g_x1x2=g_x1x2, I2=I2, g_x2x2_slow=g_x2x2_slow,
                                 c2=c2, noise=noise2) for _ in range(nbn2)]
        for j in range(nbn2):
            self.pop2[j].x2 = x2_init[j]
            self.pop2[j].y2 = y2_init[j]

        # connections between neurons
        for i in range(nbn1):
            self.pop1[i].connect_syn_pop2n(self.pop2[:])
            self.pop1[i].connect_gap(self.pop1[:])
        for j in range(nbn2):
            self.pop2[j].connect_syn_pop1n(self.pop1[:])
            self.pop2[j].connect_syn_pop2n(self.pop2[:])
            self.pop2[j].connect_gap(self.pop2[:])

        self.x1bar = np.average(x1_init)
        self.x2bar = np.average(x2_init)
        self.zbar = np.average(z_init)

    def advance_simulation(self, t_stop, x0_variable, CpES_variable):
        nbn1 = self.parameters.nbn1
        nbn2 = self.parameters.nbn2
        t_now = self.parameters.t_now
        dt = self.parameters.dt
        n = self.parameters.n

        count_samples = 0
        for ti in np.arange(t_now / dt, t_stop / dt):
            for i in range(nbn1):
                self.pop1[i].x0 = x0_variable[int(ti - t_now / dt )]
                self.pop1[i].CpES = CpES_variable[int(ti- t_now / dt )]
                self.x1_nsamples[i, count_samples], self.y1_nsamples[i, count_samples], self.z_nsamples[i, count_samples] = self.pop1[i].euler(dt, 0, self.x1bar, self.x2bar, self.zbar, ti)
                # for j in range(nbn2): # nbn1 == nbn2
                self.pop2[i].CpES = CpES_variable[int(ti- t_now / dt )]
                self.x2_nsamples[i, count_samples], self.y2_nsamples[i, count_samples] = self.pop2[i].euler(dt, 0, self.x1bar, self.x2bar, self.zbar, ti)

            self.x1bar = np.average(self.x1_nsamples[:, count_samples])
            self.x2bar = np.average(self.x2_nsamples[:, count_samples])
            self.zbar = np.average(self.z_nsamples[:, count_samples])

            self.x1bar_nsamples[count_samples] = self.x1bar
            self.x2bar_nsamples[count_samples] = self.x2bar
            self.y1bar_nsamples[count_samples] = np.average(self.y1_nsamples[:, count_samples])
            self.y2bar_nsamples[count_samples] = np.average(self.y2_nsamples[:, count_samples])
            self.zbar_nsamples[count_samples] = self.zbar

            count_samples += 1
            if count_samples == n:
                ti_n = int(ti / n)
                self.x1_plot[:, ti_n] = self.x1_nsamples.mean(axis=1)
                self.x2_plot[:, ti_n] = self.x2_nsamples.mean(axis=1)
                self.y1_plot[:, ti_n] = self.y1_nsamples.mean(axis=1)
                self.y2_plot[:, ti_n] = self.y2_nsamples.mean(axis=1)
                self.z_plot[:, ti_n] = self.z_nsamples.mean(axis=1)
                self.x1bar_plot[ti_n] = self.x1bar_nsamples.mean()
                self.x2bar_plot[ti_n] = self.x2bar_nsamples.mean()
                self.y1bar_plot[ti_n] = self.y1bar_nsamples.mean()
                self.y2bar_plot[ti_n] = self.y2bar_nsamples.mean()
                self.zbar_plot[ti_n] = self.zbar_nsamples.mean()
                self.KOP_plot[:, ti_n] = np.mean(self.KOP_nsamples, axis=1)
                count_samples = 0

        self.t_plot = np.arange(0, int((t_stop -t_now) / dt / n))
        self.t_plot_dt = np.arange(0, int((t_stop-t_now)), dt)
        self.parameters.t_now = t_stop

        evs_tot = None
        k = 0.7
        for nx, data in enumerate([self.x1_plot[: , int(t_now / dt /n):int(t_stop / dt /n)], self.x2_plot[:, int(t_now / dt /n):int(t_stop / dt /n)]]):
            for channel in range(data.shape[0]):
                mean = np.mean(data[channel,:])
                std = np.std(data[channel,:])
                thr = mean + k*std
                evs = find_threshold_indices(data[channel,:], thr)
                filtered_indices = filter_indices(evs, 60)
                # if filtered_indices[0] < 20:
                #     filtered_indices = filtered_indices[1:]
                t_evs = [channel + (nx*nbn1) for i in range(len(filtered_indices))]

                evs_tmp = np.concatenate([np.array(t_evs).reshape(-1,1), np.array(filtered_indices).reshape(-1,1)], axis=1)

                evs_tot = np.concatenate([evs_tot, evs_tmp], axis=0) if (channel + (nx*nbn1)) else evs_tmp

            if evs_tot is not None:
                self.evs = np.concatenate([self.evs, evs_tot], axis=0)

        return self.x1_plot[: , int(t_now / dt /n):int(t_stop / dt /n)], self.x2_plot[:, int(t_now / dt /n):int(t_stop / dt /n)], evs_tot


class Parameters:
    def __init__(self, args):
        self.t_tot = int(args.t_tot)
        self.CpES1 = float(args.CpES1)
        self.CpES2 = float(args.CpES2)
        self.CpCS = float(args.CpCS)
        self.m = float(args.m)
        self.r = float(args.r)
        self.s = float(args.s)
        self.x0 = float(args.x0)
        self.g_x1x1 = float(args.g_x1x1)
        self.g_x2x2 = float(args.g_x2x2)
        self.g_x1x2 = float(args.g_x1x2)
        self.g_x2x1 = float(args.g_x2x1)
        self.g_x2x1_slow = float(args.g_x2x1_slow)
        self.g_x2x2_slow = float(args.g_x2x2_slow)
        self.c2 = float(args.c2)
        self.I2 = float(args.I2)
        self.I1 = float(args.I1)
        self.nbn1 = int(args.nbn1)
        self.nbn2 = int(args.nbn2)
        self.noiseRatio = float(args.noiseRatio)
        self.noise2 = float(args.noise2)
        self.noise3 = float(args.noise3)
        self.noise1 = self.noise2 * 20
        self.dt = float(args.dt)
        self.fs = int(args.fs)
        self.n = int(args.n)
        self.t_now = 0

# #%%

# if __name__ == "__main__":

#     appendix = "test"
#     parser = argparse.ArgumentParser(description='Launch Epileptor derived population equations - ex: python populations_args --t_tot 5000 --CpES 0.1 --CpCS 0.9')
#     parser.add_argument('--t_tot', action='store', dest='t_tot', default=1000, help='time of the whole simulation (ms)')
#     parser.add_argument('--CpES1', action='store', dest='CpES1', default=0.8, help='Coupling % of electrical synapses (i.e. gap junctions) [0-1] within population 1 neurons')
#     parser.add_argument('--CpES2', action='store', dest='CpES2', default=0.8, help='Coupling % of electrical synapses (i.e. gap junctions) [0-1] within population 2 neurons')
#     parser.add_argument('--CpCS', action='store', dest='CpCS', default=1., help='Coupling % of chemical synapses [0-1]')
#     parser.add_argument('--m', action='store', dest='m', default=0.8, help='m parameter in the epileptor equations')
#     parser.add_argument('--x0', action='store', dest='x0', default=-2., help='x0 in the epileptor equations')
#     parser.add_argument('--r', action='store', dest='r', default=0.0001, help='r parameter in population 1 neurons equations')
#     parser.add_argument('--s', action='store', dest='s', default=8., help='s parameter in population 1 neurons equations, reflecting electrical impact rate on slow variable')
#     parser.add_argument('--nbn1', action='store', dest='nbn1', default=10, help='number of neurons in population 1')
#     parser.add_argument('--nbn2', action='store', dest='nbn2', default=10, help='number of neurons in population 2')
#     parser.add_argument('--g_x1x1', action='store', dest='g_x1x1', default=0.2, help='collateral synaptic (maximum) conductance between neurons from population 1')
#     parser.add_argument('--g_x2x2', action='store', dest='g_x2x2', default=0.2, help='collateral synaptic (maximum) conductance between neurons from population 2')
#     parser.add_argument('--g_x1x2', action='store', dest='g_x1x2', default=0.2, help='synaptic (maximum) conductance between neurons from population 1 to population 2')
#     parser.add_argument('--g_x2x1', action='store', dest='g_x2x1', default=0.2, help='fast synaptic (maximum) conductance between neurons from population 2 to population')
#     parser.add_argument('--g_x2x1_slow', action='store', dest='g_x2x1_slow', default=0., help='slow synaptic (maximum) conductance between neurons from population 2 to population 1')
#     parser.add_argument('--g_x2x2_slow', action='store', dest='g_x2x2_slow', default=0., help='slow synaptic (maximum) conductance between neurons from population 2 to population 1')
#     parser.add_argument('--I2', action='store', dest='I2', default=0.8, help='baseline input current in population 2 neurons')
#     parser.add_argument('--I1', action='store', dest='I1', default=3.1, help='baseline input current in population 1 neurons')
#     parser.add_argument('--c2', action='store', dest='c2', default=0.3, help='scaling factor of z injection in pop2 neurons')
#     parser.add_argument('--noise1', action='store', dest='noise1', default=0.5, help='noise amplitude that is introduced in population 1 neurons at each time step')
#     parser.add_argument('--noiseRatio', action='store', dest='noiseRatio', default=5, help='noise factor for population 1 as compared to population 2')
#     parser.add_argument('--noise2', action='store', dest='noise2', default=0.3, help='noise amplitude that is introduced in population 2 neurons at each time step')
#     parser.add_argument('--noise3', action='store', dest='noise3', default=0.0, help='noise amplitude that is introduced in slow variable at each time step')
#     parser.add_argument('--n', action='store', dest='n', default=2, help='number of samples needed to match the desired sampling frequency')
#     parser.add_argument('--dt', action='store', dest='dt', default=0.05, help='integration step of the simulation (ms)')
#     parser.add_argument('--fs', action='store', dest='fs', default=1000, help='sampling rate (Hz)')

#     args = parser.parse_args()
#     args.t_tot = 1000   # ms
#     args.dt = 0.05      # ms
#     args.fs = 500       # Hz
#     args.n = ((1/args.dt) / args.fs) * 1000

#     sim_len = args.t_tot
#     dt = args.dt

#     net = SeizureNetwork(args)
#     net.initialize_networks()

#     CpES_variable = np.zeros(int(sim_len/dt))

#     CpES_variable[0:int((sim_len/4)/dt)] = 0.2
#     CpES_variable[int((sim_len/4)/dt):int((2*sim_len/4)/dt)] = 0.2
#     CpES_variable[int(2*(sim_len/4)/dt):int((3*sim_len/4)/dt)] = 0.8
#     CpES_variable[int((3*sim_len/4)/dt):int(sim_len/dt)] = 0.8

#     x0_variable = np.zeros(int(sim_len/dt))

#     x0_variable[0:int((sim_len/4)/dt)] = -4.5
#     x0_variable[int((sim_len/4)/dt):int((2*sim_len/4)/dt)] = -2
#     x0_variable[int((2*sim_len/4)/dt):int((3*sim_len/4)/dt)] = -2
#     x0_variable[int((3*sim_len/4)/dt):int(sim_len/dt)] = -4.5

#     # start = time.time()
#     t_step = 500 # ms
#     t = np.arange(0, sim_len, int(1/args.fs*1000))
#     for ti in np.arange(0, sim_len, t_step):
#         x1_i, x2_i, evs_i = net.advance_simulation(t_stop=ti+t_step, x0_variable = x0_variable[int(ti/dt):int((ti+t_step)/dt)], CpES_variable = CpES_variable[int(ti/dt):int((ti+t_step)/dt)])
#     # end = time.time()
#     # print("loop in", (end-start))

#     # print(evs_i.shape)
#     fig = plt.figure()
#     plt.plot(t, net.x1_plot.mean(axis=0))
#     plt.plot(t, net.x2_plot.mean(axis=0))
#     plt.savefig("epileptor_orig/test")
# # %%
