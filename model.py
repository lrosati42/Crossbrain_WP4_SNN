import numpy as np
from parameters import get_reservoir_parameters, get_readout_parameters
import pyNN.neuron as pynn

class LSM_epi_nosp:
    def __init__(self, n_in: int, n_smpl: int, n_subs: int, n_res: int, n_out: int, ratio = 0.8):
        self.name = "LSM"
        self.n_in = n_in
        # self.n_smpl = n_smpl
        # self.n_subs = n_subs
        self.n_res = n_res
        self.n_exc = int(self.n_res * ratio) # excitatory neurons
        self.n_inh = self.n_res - self.n_exc # inhibitory neurons
        self.n_out = n_out

    def __str__(self):
        return f"{self.name} with {self.n_in} input neurons, {self.n_res} reservoir neurons and {self.n_out} readout neurons."
    
    def create(self):
        self.pop_in = pynn.Population(self.n_in, pynn.SpikeSourceArray(spike_times=[]))
        # self.pop_smpl = pynn.Population(self.n_smpl, pynn.SpikeSourceArray(spike_times=[]))
        # subs_id = np.random.choice(np.arange(n_smpl), size=self.n_subs, replace=False)
        # subs_id = np.array([1,7,12,18])
        # self.pop_subs = self.pop_smpl[subs_id]

        self.pop_rec = pynn.Population(self.n_res, pynn.IF_curr_exp(**get_reservoir_parameters()))

        self.pop_out = pynn.Population(self.n_out, pynn.IF_curr_exp(**get_readout_parameters()))

        # record main spike trains
        self.pop_rec.record(["spikes"])
        self.pop_out.record(["spikes"])

    def connect(self):
        # define two projections (excitatory + inhibitory) to allow signed weights
        synapse_exc = pynn.standardmodels.synapses.StaticSynapse(weight=1)
        synapse_inh = pynn.standardmodels.synapses.StaticSynapse(weight=-1)

        # Input to LSM
        self.in_rec_exc = pynn.Projection(self.pop_in, self.pop_rec,
                                pynn.AllToAllConnector(allow_self_connections=False),
                                synapse_type=synapse_exc,
                                receptor_type="excitatory")
        self.in_rec_inh = pynn.Projection(self.pop_in, self.pop_rec,
                                    pynn.AllToAllConnector(allow_self_connections=False),
                                    synapse_type=synapse_inh,
                                    receptor_type="inhibitory")
        # self.smpl_rec_exc = pynn.Projection(self.pop_subs, self.pop_rec,
        #                         pynn.AllToAllConnector(allow_self_connections=False),
        #                         synapse_type=synapse_exc,
        #                         receptor_type="excitatory")
        # self.smpl_rec_inh = pynn.Projection(self.pop_subs, self.pop_rec,
        #                             pynn.AllToAllConnector(allow_self_connections=False),
        #                             synapse_type=synapse_inh,
        #                             receptor_type="inhibitory")

        # LSM Recurrent connections
        self.proj_J_inh = pynn.Projection(self.pop_rec, self.pop_rec,
                             pynn.AllToAllConnector(allow_self_connections=False),
                             synapse_type=synapse_inh,
                             receptor_type="inhibitory")
        self.proj_J_exc = pynn.Projection(self.pop_rec, self.pop_rec,
                             pynn.AllToAllConnector(allow_self_connections=False),
                             synapse_type=synapse_exc,
                             receptor_type="excitatory")

        # Reservoir to Readout
        self.rec_out_exc = pynn.Projection(self.pop_rec, self.pop_out,
                             pynn.AllToAllConnector(allow_self_connections=False),
                             synapse_type=synapse_exc,
                             receptor_type="excitatory")
        self.rec_out_inh = pynn.Projection(self.pop_rec, self.pop_out,
                                    pynn.AllToAllConnector(allow_self_connections=False),
                                    synapse_type=synapse_inh,
                                    receptor_type="inhibitory")

    def initialize_weights(self):
        # basic initialization
        self.Jin = np.zeros((self.n_in, self.n_res))
        # self.Jsmpl = np.zeros((self.n_subs, self.n_res))
        self.Jrec = np.zeros((self.n_res, self.n_res))
        self.Jout = np.zeros((self.n_res, self.n_out))

    def clear_all(self):
        _ = self.pop_rec.get_data(clear=True)
        _ = self.pop_out.get_data(clear=True)
