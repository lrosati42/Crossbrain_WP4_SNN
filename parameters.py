# Parameters to be used in the simulation

def get_reservoir_parameters():
    res_neuron_parameters = {
            "v_rest": -65.,     # mV
            "cm": 1.,           # nF
            "tau_m": 8.,        # ms
            "tau_refrac": 0.,   # ms
            "tau_syn_E": 2.,    # ms
            "tau_syn_I": 2.,    # ms
            "i_offset": 0.,     # nA
            "v_reset": -65.,    # mV
            "v_thresh": -50.    # mV
            }
    return res_neuron_parameters

def get_readout_parameters():
    out_neuron_parameters = {
            "v_rest": -65.,     # mV
            "cm": 1.,           # nF
            "tau_m": 5.,        # ms
            "tau_refrac": 0.,   # ms
            "tau_syn_E": 2.,    # ms
            "tau_syn_I": 2.,    # ms
            "i_offset": 0.,     # nA
            "v_reset": -65.,    # mV
            "v_thresh": -50.    # mV
            }
    return out_neuron_parameters

def get_simulation_parameters():
    params = {
        "n_res": 10,                    # number of recurrent neurons
        "ratio": 0.8,                   # ratio of excitatory neurons
        "n_in": 21,                     # number of input neurons
        "n_smpl": 20,                   # number of sampled input neurons
        "n_subs": 1,                    # number of (extra) spiking input channels
        "n_out": 2,                     # number of readout neurons (classes)
        "mu_in_res": 0.5,               # weights mean from input to recurrent network
        "sigma_in_res": 0.6,            # weights std from input to recurrent network
        "mu_smpl_res": 1.,              # weights mean from input to recurrent network
        "sigma_smpl_res": 0.5,          # weights std from input to recurrent network  
        "mu_res_res": 0.,               # weights mean of recurrent connections
        "sigma_res_res": 0.3,           # weights std of recurrent connections, goes as 1/(n_main)**(1/2)
        "total_duration": 3600 * 1e3,   # total simulation duration (ms)
        "duration": 0.5 * 1e3,          # single simulation step duration (ms)
        "sample_rate": 500,             # sampling rate of the analog input signal (Hz)
        "s_pop": 1.,                    # sigma parameter of the population coding protocol
        "tau_mem": 8,                   # membrane time constant (ms)
        "tau_syn": 2,                   # synaptic time constant (ms)
        "tau_out": 5,                   # readout time constant (ms)
        "n_epochs": 10,                 # number of epochs in the training loop
        "eta": 0.003,                   # 1e-3
        "data_path": 'data/',           # data path
        "model_path": 'model_data/',    # model data path
        }
    return params