'''
Genetic algorithm for connectivity map
by Hanjia
'''

import os
import numpy as np
from random import choice, randint
from deap import base, creator, tools
import matplotlib.pyplot as plt
import time
from batch_genetic import batch_genetic

on_server = True

N_ind = 20      # number of individuals in a population
p_cx = 0.8      # cross-over probability
p_mut = 0.2     # mutation probability
max_generations = 200
mut_degrees = [0.3, 0.05]    # s.d. of mutation range (unit: times of mean)

target_corr = np.array([[0.123, 0.145, 0.112, 0.113],
                        [0.145, 0.197, 0.163, 0.193],
                        [0.112, 0.163, 0.211, 0.058],
                        [0.113, 0.193, 0.058, 0.186]])

workingdir = os.getcwd()

origin_probs = np.load('conn_probs.npy')

def create_individual():
    return origin_probs

# RMSE as fitness
def evaluate(result, target):
    t_arr = target.flatten()
    r_arr = result.flatten()
    # take out repeated elements; to be improved
    t_arr = np.concatenate((
        t_arr[0:1], t_arr[4:6], t_arr[8:11], t_arr[12:16]))
    r_arr = np.concatenate((
        r_arr[0:1], r_arr[4:6], r_arr[8:11], r_arr[12:16]))
    sum = 0.0
    cnt = 0
    fitness = 10.0
    for t, r in zip(t_arr, r_arr):
        if np.isnan(t) or np.isnan(r):
            cnt = 0 # as an error flag here
            break
        dif = (t - r) ** 2
        sum += dif
        cnt += 1
    if cnt != 0:
        fitness = np.sqrt(sum/cnt)
    return (fitness,)


def cxOnePointStr(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = randint(1, size - 1)
    indCls = type(ind1)

    new1, new2 = ind1[:cxpoint], ind2[:cxpoint]
    new1 = np.concatenate((new1, ind2[cxpoint:]))
    new2 = np.concatenate((new2, ind1[cxpoint:]))

    return indCls(new1), indCls(new2)


def mutSNP1(ind, p):
    l23 = np.arange(0, 4)
    l4 = np.arange(4, 7)
    l5 = np.arange(7, 10)
    l6 = np.arange(10, 13)
    assert (0 <= p <= 1)
    new = ind
    for i, row in enumerate(new):
        for j, item in enumerate(row):
            if np.random.random() <= p:
                if i in l23:    # only when target is in L2/3
                    if j in l23:
                        mut_sd = mut_degrees[0]
                    else:
                        mut_sd = mut_degrees[1]
                    new_item = item + mut_sd * item * (np.random.randn())

                    # boundaries
                    origin = origin_probs[i, j]
                    if new_item > origin*1.5:
                        new_item = origin*1.5
                    elif new_item < origin*0.5:
                        new_item = origin*0.5

                    new[i][j] = new_item
    return type(ind)(new)


def do_and_check(survivors, g):
    if on_server:
        # divide into groups of 10
        round_num = 10
        for i in range(int(len(survivors) / round_num)):
            # do the simulations
            map_ids = np.arange(i * round_num, (i + 1) * round_num)
            batch_genetic(survivors[i * round_num:(i + 1) * round_num], g, map_ids)
            fin_flag = False  # finish flag
            t0 = time.time()
            # check results
            while fin_flag is False:
                time.sleep(60)
                fin_flag = True
                for map_id in map_ids:
                    if os.path.isfile(
                                workingdir +
                                '/output/g={0:02d}_ind={1:02d}/coef_arr.npy'.format(g, map_id)
                    ) is False:
                        fin_flag = False
                # break if this generation takes too long
                if time.time() - t0 > (4 * 3600):
                    break
    else:
        for i in range(len(survivors)):
            datapath = workingdir + '/output/g={0:02d}_ind={1:02d}/'.format(g, i)
            if os.path.isdir(datapath) is False:
                os.mkdir(datapath)
            np.save(datapath + 'coef_arr.npy',
                    0.1 * np.random.randn(4, 4) + 0.1)
            # np.save(datapath + 'coef_arr.npy', np.full((4,4), np.nan))


# fitness function should minimize the difference between present and target str
creator.create('FitMin', base.Fitness, weights=(-1,))
# individual is a list (conn map)
creator.create('Individual', list, fitness=creator.FitMin)  # , n=len(target))

# register functions needed for initialization, evaluation etc.
box = base.Toolbox()
box.register('create_ind', create_individual)
box.register('ind', tools.initIterate, creator.Individual, box.create_ind)
box.register('pop', tools.initRepeat, list, box.ind)

box.register('evaluate', evaluate)
box.register('crossover', cxOnePointStr)
box.register('mutate', mutSNP1)
box.register('select', tools.selTournament, tournsize=3)

### INITIALIZATION
population = box.pop(n=N_ind)

# ### EVOLUTION
g = 0
fits = [10 for i in population]
fitness_evolved = np.zeros((max_generations, 5))
best5_inds_evolved = np.zeros((max_generations, 5))
# Target: fitness (RMSE) of pre- vs. post-learning exp. data ~= 0.06
# evolved fitness should be at least smaller than this level
while min(fits) > 0.005 and g < max_generations:
    ## SELECTION
    survivors = box.select(population, len(population))

    ## GENETIC OPERATIONS
    # crossing-over
    half = int(len(survivors) / 2)
    chances = np.random.random(size=half)
    chances = np.where(chances <= p_cx)[0]
    for i in chances:
        new1, new2 = box.crossover(survivors[i], survivors[i + half])
        survivors[i] = new1
        survivors[i + half] = new2
        # new1 and new2 are new instances of Individual class, so there is no
        # need to delete or invalidate their fitness values

    # mutation
    for i, ind in enumerate(survivors):
        survivors[i] = box.mutate(ind, p_mut)

    # SIMULATION
    do_and_check(survivors, g)

    # EVALUATION
    for i, ind in enumerate(survivors):
        corr_file = os.getcwd() + '/output/g={0:02d}_ind={1:02d}/'.format(g, i) + 'coef_arr.npy'
        if os.path.isfile(corr_file):
            result_arr = np.load(corr_file)
        else:
            result_arr = np.full((4, 4), np.nan)
        ind.fitness.values = box.evaluate(result_arr, target_corr)

    population[:] = survivors
    fits = [i.fitness.values[0] for i in population]
    print('generation {} min. FitMin is {}'.format(g, min(fits)))
    print('fitness values = {}'.format(fits))

    # save fitness values
    fitness_evolved[g, :] = np.array(
        [population[i].fitness.values for i in np.argsort(fits)[:5]]).reshape(
        5)
    np.save(
        workingdir + '/output/fitness_evolved_g{:02d}.npy'.format(g),
        fitness_evolved)

    # save evolved better individuals
    best5_inds_evolved[g, :] = np.arange(0, 20)[np.argsort(fits)[:5]]
    np.save(
        workingdir + '/output/inds_evolved_g{:02d}.npy'.format(g),
        best5_inds_evolved)

    # delete .gdf files to save space
    for i in range(20):
        data_dir = os.getcwd() + '/output/g={0:02d}_ind={1:02d}/'.format(g, i) + 'data/'
        for item in os.listdir(data_dir):
            if item.endswith('.gdf'):
                os.remove(data_dir + item)

    g += 1

plt.figure()
plt.plot(np.arange(g), np.sqrt(fitness_evolved[:g, :]/10), 'b.')
plt.hlines(0.06, 0, g + 10, 'k', linestyles='--', label='pre vs. post RMSE')
plt.hlines(0, 0, g + 10, 'w', linestyles='--')
plt.legend()
plt.xlabel('Number of generations')
plt.ylabel('Fitness (RMSE)')
plt.title("Evolution of 5 best individuals' fitness")
plt.tight_layout()
plt.savefig(workingdir + '/output/genetic_hj.png')
