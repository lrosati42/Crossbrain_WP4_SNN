import numpy as np
from tqdm import trange
import pyNN.neuron as pynn
from scipy.spatial import distance
from _static.utils import alpha, filtering, set_weights, spikes_count
from _static import optimizer

class Suptrain_data:
    def __init__(self):
        self.name = "Self supervised training"

    def __str__(self):
        return f"{self.name} with Adam optimizer."

    def training_loop(self, params: dict, net, coding, decoding, epinet, mean, x0, CpES, decoder, n_epochs):
        print(f'Training for {n_epochs} epochs')
        # hyperparameters
        duration = params["duration"]
        smpl_freq = params["sample_rate"]
        timepoints = int(duration * smpl_freq / 1e3)
        dt = (1 / smpl_freq) * 1e3
        sim_steps = int(duration/epinet.parameters.dt)

        tau_out = params["tau_out"]
        alpha_out = alpha(dt, tau_out)

        # optimizer
        lr = params["eta"]
        self.opt_out = optimizer.Adam(alpha=lr, drop=0.9, drop_time=500000)
        # metrics
        self.loss = np.zeros(n_epochs)
        epochs = trange(n_epochs)
        self.out_x0 = np.zeros(n_epochs)
        self.out_CpES = np.zeros(n_epochs)

        # TRAINING LOOP
        for e in epochs:
            x0_array = np.array([x0[e]]*sim_steps)
            CpES_array = np.array([CpES[e]]*sim_steps)

            ##### SET INPUT SPIKETIMES #####
            input_signal = mean[e*timepoints:(e+1)*timepoints]
            input_times = coding.get_coding(input_signal, dt)
            net.pop_in.set(spike_times = input_times)

            ##### RUN #####
            pynn.run_until(duration)
            # retrieve data
            data_out = net.pop_out.get_data(clear=True)
            data = net.pop_rec.get_data(clear=True)
            spikes = data.segments[-1].spiketrains

            # reset pyNN state
            pynn.reset()

            ##### DATA FILTERING #####
            spikes_filtered_tout = filtering(spikes, alpha_out, timepoints)

            ##### METRICS #####
            g_x0, g_CpES = x0_array.mean(), CpES_array.mean()
            self.out_x0[e] = g_x0
            self.out_CpES[e] = g_CpES
            a_x0, a_CpES = decoding.decode(data_out)
            self.loss[e] = distance.euclidean([g_x0, g_CpES], [a_x0, a_CpES])

            ##### WEIGHTS UPDATE #####
            diff = np.stack(([g_x0-a_x0]*timepoints, [g_CpES-a_CpES]*timepoints))
            dJout = spikes_filtered_tout @ diff.T
            net.Jout = (self.opt_out.step(net.Jout.T, dJout.T)).T

            ##### SET READOUT WEIGHTS #####
            set_weights(net.Jout, net.rec_out_inh, net.rec_out_exc)

            ##### STATUS #####
            epochs.set_postfix({'loss': self.loss[e], 'relative loss': self.loss[e]/self.loss[0], 'spikes': spikes_count(data.segments[-1].spiketrains).sum()})

        print(f'Training mean loss: {self.loss.mean():.3f}')
