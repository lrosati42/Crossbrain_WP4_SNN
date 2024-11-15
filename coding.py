import numpy as np
from elephant import statistics
from elephant import spike_train_generation as stg
import quantities as pq
import pyNN.neuron as pynn
from _static.utils import spike_prob, spike_matrix, spike_times

class no_coding:
    def __init__(self, n_in: int, n_classes: int):
        self.name = "Simple synthetic data"
        self.n_in = n_in
        if self.n_in != 1:
            print("More than 1 input neuron in the no coding regime!")
        self.n_classes = n_classes
        fmin = 55 # Hz
        fmax = 150 # Hz
        self.freqs = np.linspace(fmin, fmax, self.n_classes)
        self.dts = 1 / self.freqs # s

        self.sts = []
        for f in range(len(self.freqs)):
            self.sts.append(np.round(np.arange(self.dts[f] * 1e3, 1000, self.dts[f] * 1e3))) # times in ms HANDWRITTEN

    def __str__(self):
        return f"{self.name} with {self.n_in} neurons and {self.n_classes} classes (spike frequencies)."

    def get_coding(self, l: int):
        # get input neuron spike times
        if not l:
            return ([])
        else:
            return self.sts[l-1]

class synth_sip:
    def __init__(self, n_in: int, n_classes: int, LIF_params: dict):
        self.name = "Fancy synthetic data"
        self.n_in = n_in
        if self.n_in != 1:
            print("More than 1 collecting neuron in the SIP data generator!")
        self.n_classes = n_classes
        self.n_synth_exc = 100
        self.n_synth_inh = 100
        self.rate_exc = 100  # Hz
        self.rate_inh = 50   # Hz
        self.jitter_exc = 5  # ms

        # define pyNN population that will provide synthetic spiketimes
        self.pop_synth_E = pynn.Population(self.n_synth_exc, pynn.SpikeSourceArray(spike_times=[]))
        self.pop_synth_I = pynn.Population(self.n_synth_inh, pynn.SpikeSourceArray(spike_times=[]))
        # define pyNN LIF neuron that collects spikes from the SIP process
        self.collect_LIF = pynn.Population(self.n_in, pynn.IF_cond_alpha(**LIF_params))
        self.collect_LIF.record(["spikes"])
        # define two projections (excitatory + inhibitory) to allow signed weights
        cond_syn_exc = pynn.standardmodels.synapses.StaticSynapse(weight=1e-3) # 1nS
        cond_syn_inh = pynn.standardmodels.synapses.StaticSynapse(weight=3.4e-3) # 3.4nS
        # connect spike generators to LIF collector
        self.synth_coll_exc = pynn.Projection(self.pop_synth_E, self.collect_LIF,
                             pynn.AllToAllConnector(allow_self_connections=False),
                             synapse_type=cond_syn_exc,
                             receptor_type="excitatory")
        self.synth_coll_inh = pynn.Projection(self.pop_synth_I, self.collect_LIF,
                             pynn.AllToAllConnector(allow_self_connections=False),
                             synapse_type=cond_syn_inh,
                             receptor_type="inhibitory")

    def __str__(self):
        return f"{self.name} based on SIP model for correlated spike patterns with {self.n_in} collecting neurons and {self.n_classes} classes."

    def get_coding(self, l: int):
        if not l:
            # get input neuron spike times
            sip = stg.single_interaction_process(rate=self.rate_exc*pq.Hz, coincidence_rate=0.*pq.Hz, t_stop=1000*pq.ms, n_spiketrains=self.n_synth_exc, jitter = self.jitter_exc *pq.ms)
            nosip = stg.single_interaction_process(rate=self.rate_inh*pq.Hz, coincidence_rate=0.*pq.Hz, t_stop=1000*pq.ms, n_spiketrains=self.n_synth_inh, jitter= self.jitter_exc *pq.ms)            
        else:
            # get input neuron spike times
            sip = stg.single_interaction_process(rate=self.rate_exc*pq.Hz, coincidence_rate=self.rate_exc*pq.Hz, t_stop=1000*pq.ms, n_spiketrains=self.n_synth_exc, jitter = self.jitter_exc *pq.ms, coincidences = 'deterministic')
            nosip = stg.single_interaction_process(rate=self.rate_inh*pq.Hz, coincidence_rate=0.*pq.Hz, t_stop=1000*pq.ms, n_spiketrains=self.n_synth_inh, jitter= self.jitter_exc *pq.ms)
        self.pop_synth_E.set(spike_times = sip)
        self.pop_synth_I.set(spike_times = nosip)
        pynn.run(1000)
        data = self.collect_LIF.get_data(clear=True)
        pynn.reset()
        return (data.segments[-1].spiketrains[-1].magnitude)

class population_coding:
    def __init__(self, n_in: int, s_pop: float):
        self.name = "Population coding"
        self.n_in = n_in
        self.s_pop = s_pop
        interval = 1 / (self.n_in - 1)
        self.sigma_pop = interval * self.s_pop
        xmax = 1
        xmin = 0
        self.xs = np.linspace(xmin, xmax, self.n_in)

    def __str__(self):
        return f"{self.name} with {self.n_in} neurons and sigma = {self.s_pop:.1f} dx."
    
    def get_coding(self, signal, timestep: float):
        # get coding neurons spike probabilities
        sp = spike_prob(signal, self.xs, self.sigma_pop)
        # get matrix of spikes
        sm = spike_matrix(sp)
        # extract times from spikes
        return spike_times(sm, timestep)

class FR_decoding:
    def __init__(self, fmin = 0, fmax = 100):
        self.name = "Output rate coding"
        self.fmax = fmax
        self.fmin = fmin

    def __str__(self):
        return f"{self.name} with frequency between {self.fmin} and {self.fmax} Hz."
    
    def get_fr(self, data_out):
        fr_0 = statistics.mean_firing_rate(data_out.segments[-1].spiketrains[0]).magnitude*1e3 # Hz
        fr_1 = statistics.mean_firing_rate(data_out.segments[-1].spiketrains[1]).magnitude*1e3 # Hz
        return fr_0, fr_1

    def get_params(self, fr_x0, fr_cp):
        x0 = (fr_x0 - self.fmin) / (self.fmax - self.fmin) if fr_x0<self.fmax else 1
        cp = (fr_cp - self.fmin) / (self.fmax - self.fmin) if fr_cp<self.fmax else 1
        return x0, cp

    def decode(self, data_out):
        fr_0, fr_1 = self.get_fr(data_out)
        x0, cp = self.get_params(fr_0, fr_1)
        return x0, cp