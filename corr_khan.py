import numpy as np
import matplotlib
import microcircuit_tools as tools
# matplotlib.rcParams['font.size'] = 30.0

labels = ['E', 'PV', 'SOM', 'VIP']

def evaluate(result, target):
    fitness = 0
    t_arr = target.flatten()
    r_arr = result.flatten()
    # desert repeated elements; to be improved
    t_arr = np.concatenate((
        t_arr[0:4], t_arr[5:8], t_arr[10:12], t_arr[15:16]))
    r_arr = np.concatenate((
        r_arr[0:4], r_arr[5:8], r_arr[10:12], r_arr[15:16]))
    for t, r in zip(t_arr, r_arr):
        if np.isnan(t) or np.isnan(r):
            fitness = 10.0
            break
        dif = (t - r) ** 2
        fitness += dif
    return (fitness,)

# pre-learning
arr_pre = np.array([[0.123, 0.145, 0.112, 0.113],
                [0.145, 0.197, 0.163, 0.193],
                [0.112, 0.163, 0.211, 0.058],
                [0.113, 0.193, 0.058, 0.186]])

# post-learning
arr_post = np.array([[0.075, 0.092, 0.035, 0.092],
                [0.092, 0.144, 0.036, 0.151],
                [0.035, 0.036, 0.204, 0.000],
                [0.092, 0.151, 0.000, 0.176]])


fitness = evaluate(arr_post, arr_pre)[0]
print('fitness (pre vs. post) = {:.3f}'.format(fitness))

tools.interaction_barplot(arr_post, -0.1, 0.25, labels, 'mean corr coef')
