import nest
import matplotlib.pyplot as plt

n_model = 'iaf_psc_exp'
defaults = nest.GetDefaults(n_model)
syn_weight = 20.0

syn_dict_d = {
    'model': 'tsodyks_synapse',
    'U': 0.75,
    'tau_fac': 0.01,
    'tau_psc': defaults['tau_syn_ex'],
    'tau_rec': 800.0,
    'weight': syn_weight
}

syn_dict_f = {
    'model': 'tsodyks_synapse',
    'U': 0.1,
    'tau_fac': 200.0,
    'tau_psc': defaults['tau_syn_ex'],
    'tau_rec': 0.01,
    'weight': syn_weight
}

n1 = nest.Create(n_model)
n2 = nest.Create(n_model)
nest.SetStatus(n2, {'I_e': 500.0})
m1 = nest.Create('multimeter')
nest.SetStatus(m1, {"withtime": True, "record_from": ["V_m"]})

nest.Connect(n2, n1, syn_spec = syn_dict_f)

nest.Connect(m1, n1)

nest.Simulate(1000.0)

dmm = nest.GetStatus(m1)[0]
Vms = dmm["events"]["V_m"]
ts = dmm["events"]["times"]
plt.plot(ts, Vms)
plt.ylim((-70.0, -69.9))
plt.show()

