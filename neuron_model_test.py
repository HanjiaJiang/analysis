import nest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 20.0


n_model = 'iaf_psc_exp'
defaults = nest.GetDefaults(n_model)
syn_weight = 20.0
sim_time = 500.0

# spike generator
isi = 30.0
spike_times = np.arange(isi, sim_time, isi)
spk = nest.Create('spike_generator')
nest.SetStatus(spk, {'spike_times': spike_times})

syn_dict_n = {
    'weight': syn_weight
}

syn_dict_d = {
    'model': 'tsodyks_synapse',
    'U': 0.5,
    'tau_fac': 100.0,
    'tau_psc': defaults['tau_syn_ex'],
    'tau_rec': 200.0, #800.0,
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
nd2 = nest.Create('parrot_neuron')
# nest.SetStatus(nd2, {'I_e': 450.0})

md = nest.Create('multimeter')
nest.SetStatus(md, {"withtime": True, "record_from": ["V_m"]})

nest.Connect(spk, nd2)
nest.Connect(nd2, nd1, syn_spec = syn_dict_d)

nest.Connect(md, nd1)

# facilitated
nf1 = nest.Create(n_model)
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
plt.ylim((-70.0, -69.8))
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
plt.ylim((-70.0, -69.8))
plt.xlabel('t (ms)')
plt.ylabel('V')
plt.title('facilitated (E to SOM)')
plt.savefig('stp_facilitated.png')
plt.close()
