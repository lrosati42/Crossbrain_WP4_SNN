import numpy as np
from tqdm import trange
import pyNN.neuron as pynn
from scipy.spatial import distance
from par_space_dynamics import denorm_params

class Suptest_data:
    def __init__(self):
        self.name = "Test"

    def __str__(self):
        return f"{self.name}."

    def test_loop(self, params: dict, net, coding, decoding, epinet, mean, x0, CpES, decoder, n_epochs):
        print(f'Testing for {n_epochs} epochs')
        # hyperparameters
        duration = params["duration"]
        smpl_freq = params["sample_rate"]
        timepoints = int(duration * smpl_freq / 1e3)
        dt = (1 / smpl_freq) * 1e3
        sim_steps = int(duration/epinet.parameters.dt)

        # metrics
        self.loss = np.zeros(n_epochs)
        epochs = trange(n_epochs)
        self.out_x0 = np.zeros(n_epochs)
        self.out_CpES = np.zeros(n_epochs)

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

            # reset pyNN state
            pynn.reset()

            ##### METRICS #####
            g_x0, g_CpES = x0_array.mean(), CpES_array.mean()
            a_x0, a_CpES = decoding.decode(data_out)
            x0_o, CpES_o = denorm_params(a_x0, a_CpES)
            self.out_x0[e] = x0_o
            self.out_CpES[e] = CpES_o
            self.loss[e] = distance.euclidean([g_x0, g_CpES], [a_x0, a_CpES])

            ##### STATUS #####
            epochs.set_postfix({'loss': self.loss[e], 'x0': x0_o, 'CpES': CpES_o})

        print(f'Test mean loss: {self.loss.mean():.3f}')
