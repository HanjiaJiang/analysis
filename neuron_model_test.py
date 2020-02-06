import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 20.0

# to be done
# exc = {
#     # Reset membrane potential of the neurons (in mV).
#     'E_L': -63.3,
#     # Threshold potential of the neurons (in mV).
#     'V_th': -41.0,
#     # Membrane potential after a spike (in mV).
#     'V_reset': -67.0,  # -65.0,
#     # Membrane capacitance (in pF).
#     'C_m': 322.0,
#     # Membrane time constant (in ms).
#     'tau_m': 13.0,
#     # Time constant of postsynaptic excitatory currents (in ms).
#     'tau_syn_ex': 1.74,  # 1.9,  # Allen mouse #1.74, #1.0, # 0.5,
#     # Time constant of postsynaptic inhibitory currents (in ms).
#     'tau_syn_in': 4.6,  # 2.9,  # Allen mouse #4.6, #2.0, # 0.5,
# }

pv = {
    # Reset membrane potential of the neurons (in mV).
    'E_L': -66.8,
    # Threshold potential of the neurons (in mV).
    'V_th': -40.5,
    # Membrane potential after a spike (in mV).
    'V_reset': -67.0,  # -65.0,
    # Membrane capacitance (in pF).
    'C_m': 86.2,
    # Membrane time constant (in ms).
    'tau_m': 3.6,
    # Time constant of postsynaptic excitatory currents (in ms).
    'tau_syn_ex': 1.74,  # 1.9,  # Allen mouse #1.74, #1.0, # 0.5,
    # Time constant of postsynaptic inhibitory currents (in ms).
    'tau_syn_in': 4.6,  # 2.9,  # Allen mouse #4.6, #2.0, # 0.5,
}

som = {
    # Reset membrane potential of the neurons (in mV).
    'E_L': -61.6,
    # Threshold potential of the neurons (in mV).
    'V_th': -40.3,
    # Membrane potential after a spike (in mV).
    'V_reset': -67.0,  # -65.0,
    # Membrane capacitance (in pF).
    'C_m': 134.0,
    # Membrane time constant (in ms).
    'tau_m': 11.8,
    # Time constant of postsynaptic excitatory currents (in ms).
    'tau_syn_ex': 1.74,  # 1.9,  # Allen mouse #1.74, #1.0, # 0.5,
    # Time constant of postsynaptic inhibitory currents (in ms).
    'tau_syn_in': 4.6,  # 2.9,  # Allen mouse #4.6, #2.0, # 0.5,
}

n_model = 'iaf_psc_exp'
defaults = nest.GetDefaults(n_model)
syn_weight = 20.0
sim_time = 500.0

# spike generator
isi = 20.0
spike_times = np.arange(0.1, sim_time, isi)
spk = nest.Create('spike_generator')
nest.SetStatus(spk, {'spike_times': spike_times})

syn_dict_n = {
    'weight': syn_weight
}

syn_dict_d = {
    'model': 'tsodyks_synapse',
    'U': 0.75,
    'tau_fac': 0.0,
    'tau_psc': defaults['tau_syn_ex'],
    'tau_rec': 100.0, #800.0,
    'weight': syn_weight
}

syn_dict_f = {
    'model': 'tsodyks_synapse',
    'U': 0.5,
    'tau_fac': 200.0,
    'tau_psc': defaults['tau_syn_ex'],
    'tau_rec': 0.01,
    'weight': syn_weight
}


# depressed
nd1 = nest.Create(n_model)
nest.SetStatus(nd1, pv)
nd2 = nest.Create('parrot_neuron')
# nest.SetStatus(nd2, {'I_e': 450.0})

md = nest.Create('multimeter')
nest.SetStatus(md, {"withtime": True, "record_from": ["V_m"]})

nest.Connect(spk, nd2)
nest.Connect(nd2, nd1, syn_spec = syn_dict_d)

nest.Connect(md, nd1)

# facilitated
nf1 = nest.Create(n_model)
nest.SetStatus(nf1, som)
nf2 = nest.Create('parrot_neuron')
# nest.SetStatus(nf2, {'I_e': 450.0})

mf = nest.Create('multimeter')
nest.SetStatus(mf, {"withtime": True, "record_from": ["V_m"]})

nest.Connect(spk, nf2)
nest.Connect(nf2, nf1, syn_spec = syn_dict_f)

nest.Connect(mf, nf1)

nest.Simulate(sim_time)

# "depressed" data
dmm = nest.GetStatus(md)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]

plt.plot(ts, Vms, 'g')
plt.tight_layout()
plt.ylim((pv['E_L'], pv['E_L'] + 0.2))
plt.xlabel('t (ms)')
plt.ylabel('V')
plt.title('depressed (E to PV)')
plt.savefig('stp_depressed.png')
plt.close()

# "facilitated" data
dmm = nest.GetStatus(mf)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
plt.plot(ts, Vms, 'b')
plt.tight_layout()
plt.ylim((som['E_L'], som['E_L'] + 0.2))
plt.xlabel('t (ms)')
plt.ylabel('V')
plt.title('facilitated (E to SOM)')
plt.savefig('stp_facilitated.png')
plt.close()
