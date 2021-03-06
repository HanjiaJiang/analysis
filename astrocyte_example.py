import numpy as np
from matplotlib import pyplot as plt
import nest

nrn_model = "aeif_cond_alpha_astro"
astro_model = "astrocyte"
conn_nrn_astro = "tsodyks_synapse"
conn_nrn_nrn = "tsodyks_synapse"
conn_astro_nrn = "sic_connection"

# Create pre- and postsynaptic neurons and an astrocyte
send_nrn = nest.Create(nrn_model, 100)
rec_nrn = nest.Create(nrn_model, 100)
astro = nest.Create(astro_model, 100)

w_n2n = np.random.normal(200.0, 20.0, 100)
w_n2a = np.random.normal(60.0, 6.0, 100)
w_a2n = np.random.normal(2.11, 0.2, 100)

conn_dict = {'rules': 'one_to_one'}

# A simple connectivity scheme.
# The connections to the astrocyte need to be exact copies of presynaptic
# connections
nest.Connect(send_nrn, rec_nrn, conn_spec=conn_dict, syn_spec={"model": conn_nrn_nrn, 'weight': w_n2n})
nest.Connect(send_nrn, astro, conn_spec=conn_dict, syn_spec={"model": conn_nrn_astro, 'weight': w_n2a})
nest.Connect(astro, rec_nrn, conn_spec=conn_dict,
             syn_spec={"model": conn_astro_nrn, 'weight': w_a2n})

# Create recording and stimulation devices
pre_multimeter = nest.Create(
    "multimeter", params={"record_from": ["V_m"], "withtime": True}
)
nest.Connect(pre_multimeter, send_nrn)
post_multimeter = nest.Create(
    "multimeter", params={"record_from": ["V_m"], "withtime": True}
)
nest.Connect(post_multimeter, rec_nrn)
curr_gen = nest.Create("dc_generator", params={"start": 10.0, "stop": 1000.0, "amplitude": 800.0})
nest.Connect(curr_gen, send_nrn)
astro_meter = nest.Create("multimeter", params={"record_from": ["IP3", "Ca_astro"], "withtime": True})
nest.Connect(astro_meter, astro)

nest.Simulate(2000.0)

mm_data = nest.GetStatus(pre_multimeter)[0]["events"]
times = mm_data["times"]
voltages = mm_data["V_m"]

post_data = nest.GetStatus(post_multimeter)[0]["events"]

ip3_data = nest.GetStatus(astro_meter)[0]["events"]

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(times, voltages)
plt.plot(post_data['times'], post_data['V_m'])
plt.subplot(3, 1, 2)
plt.plot(ip3_data["times"], ip3_data["IP3"])
plt.subplot(3, 1, 3)
plt.plot(ip3_data["times"], ip3_data["Ca_astro"])
plt.show()
