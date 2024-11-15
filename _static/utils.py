import numpy as np
from scipy.stats import norm
from csv import Sniffer
from elephant.conversion import BinnedSpikeTrain
np.random.seed()

def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def get_weights(ee, ei, ie, ii):
    w_ee = ee.get('weight', format='array')
    w_ei = ei.get('weight', format='array')
    w_ie = ie.get('weight', format='array')
    w_ii = ii.get('weight', format='array')

    return w_ee, w_ei, w_ie, w_ii

def deconstruct_w(weights):
    n_rec = weights.shape[0]
    n_exc = int(n_rec*0.8)
    n_inh = n_rec - n_exc

    w_e_e = weights[:n_exc, :n_exc]
    w_e_i = weights[:n_exc, -n_inh:]
    w_i_e = weights[-n_inh:, :n_exc]
    w_i_i = weights[-n_inh:, -n_inh:] 

    return w_e_e, w_e_i, w_i_e, w_i_i

def deconstruct_w_out(weights):
    n_rec = weights.shape[0]
    n_exc = int(n_rec*0.8)
    n_inh = n_rec - n_exc

    w_e_o = weights[:n_exc, :]
    w_i_o = weights[-n_inh:, :]

    return w_e_o, w_i_o

def set_weights(weights, projection_inh, projection_exc, w_max=1000):
    # limit weights to hw boundaries
    weights = weights.clip(-w_max, w_max)

    # integer_weights = np.round(weights).astype(int)
    w_exc = weights * (weights >= 0)
    w_inh = weights * (weights < 0)

    projection_inh.set(weight=w_inh)
    projection_exc.set(weight=w_exc)

def set_positive_weights(weights, projection_exc, w_max=63):
    # limit weights to hw boundaries
    weights = weights.clip(0, w_max)

    # integer_weights = np.round(weights).astype(int)
    w_exc = weights * (weights >= 0)

    projection_exc.set(weight=w_exc)

def set_negative_weights(weights, projection_inh, w_max=63):
    # limit weights to hw boundaries
    weights = weights.clip(-w_max, 0)

    # integer_weights = np.round(weights).astype(int)
    w_inh = weights * (weights <= 0)

    projection_inh.set(weight=w_inh)

def set_rec_weights(weights, exc_exc, exc_inh, inh_exc, inh_inh):
    w_e_e, w_e_i, w_i_e, w_i_i = deconstruct_w(weights)

    set_positive_weights(w_e_e, exc_exc)
    set_positive_weights(w_e_i, exc_inh)
    set_negative_weights(w_i_e, inh_exc)
    set_negative_weights(w_i_i, inh_inh)

def set_out_weights(weights, exc_out, inh_out):
    w_e_o, w_i_o = deconstruct_w_out(weights)

    set_positive_weights(w_e_o, exc_out)
    set_negative_weights(w_i_o, inh_out)

def alpha(dt, tau):
    return (1 - np.exp(-dt/tau))

def spike_prob(x, xi, sigma):
    prob = np.zeros((len(xi), len(x)))
    for i in range(prob.shape[0]):
        prob[i] = norm.pdf(x, xi[i], sigma)

    return prob/prob.max()

def spike_matrix(probs):
    matrix = np.zeros_like(probs.T)
    for i in range(matrix.shape[0]): # cycle on time intervals
        while len(matrix[i].nonzero()[0]) < 1:
            for j in range(matrix.shape[1]): # cycle on neurons
                if np.random.random_sample() < probs.T[i][j]:
                    matrix[i][j] = 1

    return matrix.T

def spike_times(matrix, timestep):
    spike_trains = []
    for n in range(matrix.shape[0]):
        if n in matrix.nonzero()[0]:
            index = np.where(matrix.nonzero()[0] == n)
            spike_trains.append(matrix.nonzero()[1][index] * timestep)
        else:
            spike_trains.append(np.array([]))

    return spike_trains

def spikes_count(spikes):
    return (BinnedSpikeTrain(spikes, n_bins=1, tolerance=None).to_array())

def binning(spikes, n_steps):
    return (BinnedSpikeTrain(spikes, n_bins=n_steps, tolerance=None).to_array(dtype=float))

def filtering(spikes, a, n_steps): # normalized
    spikes = binning(spikes, n_steps)
    spikes_f = spikes.copy()
    for t in range(1, n_steps):
        spikes_f.T[t] = a * spikes.T[t] + (1 - a) * spikes_f.T[t-1]

    return spikes_f/a

def filtering_nobin(spikes, a, n_steps): # normalized
    spikes_f = spikes.copy()
    for t in range(1, n_steps):
        spikes_f.T[t] = a * spikes.T[t] + (1 - a) * spikes_f.T[t-1]

    return spikes_f/a

def MSE(act1, act2):
    if (np.isnan(act1.min())) or (np.isnan(act2.min())):
        return 4
    else:
        return np.sum(np.square(act1 - act2))/len(act1)

def MSE_offT(act1, act2, offT=1):
    return MSE(act1[offT:], act2[offT:])

def get_spiketimes(evs):
    offT = 10 # TODO handwritten
    if evs is None:
        return np.array([]*20)
    else:
        ste = []
        sti = []
        for n in range(offT):
            ide = np.where(evs.T[0] == n)[0]
            idi = np.where(evs.T[0] == n+offT)[0]
            ste.append(evs.T[1][ide]) # spike times in in ms
            sti.append(evs.T[1][idi]) # spike times in in ms

        ste = np.array(ste, dtype=object)
        sti = np.array(sti, dtype=object)
        
    st = []
    for i in range(len(ste)+len(sti)):
        if i < len(ste):
            st.append(ste[i])
        else:
            st.append(sti[i-len(ste)])
    return st

def binarize_spiketrain(spiketrain, timebin, n_steps):
    binarized = np.zeros((len(spiketrain), n_steps))
    for i in range(n_steps):
        for j in range(len(spiketrain)):
            for s in range(len(spiketrain[j])):
                if (i*timebin <= spiketrain[j][s] < i*timebin+timebin):
                    binarized[j][i] = 1
    return binarized

def pop_coding(signal, xs, sigma_pop, timestep):
    # get coding neurons spike probabilities
    sp = spike_prob(signal, xs, sigma_pop)
    # get matrix of spikes
    sm = spike_matrix(sp)
    # extract times from spikes
    return spike_times(sm, timestep)

def has_header(file): 
    with open(file, 'r') as csvfile:
        sniffer = Sniffer()
    return sniffer.has_header(csvfile.read())

def bitrep(x, n_bits=6):
    return np.array(list(f"{x:0{n_bits}b}"), dtype=int)

def discretize_weights(J: np.array, n_bits: int, rounding: int, mode = 'linear', jmin = None, jmax = None, bitrep = False, LSB = False):
    n_values = (2**n_bits -1)
    jmin = jmin if (jmin != None) else np.abs(J).min()
    jmax = jmax if (jmax != None) else np.abs(J).max()

    if jmin > jmax:
        print('Wrong weight range.')
        return None

    if mode == 'linear':
        values = np.linspace(jmin, jmax, n_values) # linearly spaced intervals
    else:
        print('Spacing mode not implemented.')
        return None

    values = np.round(values, rounding)
    wLSB = values[values > 0].min()
    values = np.unique(np.concatenate([-np.flip(values), values]))
    bins = np.digitize(J, values)
    bins[bins == values.size] = np.ones_like(bins[bins == values.size]) * (values.size -1) # workaround for digitize out of borders

    out = values[bins.flatten()].reshape(J.shape)

    if bitrep and LSB:
        return out, bins, wLSB
    elif bitrep or LSB:
        return out, bins if bitrep else out, wLSB
    else:
        return out