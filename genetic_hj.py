'''
Genetic algorithm for connectivity map
by Hanjia

Idea taken from:
https://www.electricmonk.nl/log/2011/09/28/evolutionary-algorithm-evolving-hello-world/
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
p_cx = 0.8      # probability of cross-over
p_mut = 0.2     # probability of mutation
max_generations = 40
mut_degrees = [0.3, 0.0]    # s.d. of mutation range (unit: times of mean)

origin_map = np.array([
    [0.0872, 0.3173, 0.4612, 0.0448, 0.1056, 0.4011, 0.0374, 0.0234, 0.09,
     0.1864, 0., 0., 0.],
    [0.3763, 0.3453, 0.2142, 0.0683, 0.0802, 0.012, 0.0257, 0.0257, 0.1937,
     0.2237, 0.0001, 0.0001, 0.0062],
    [0.2288, 0.1342, 0.1242, 0.2618, 0.0033, 0.0097, 0.0363, 0.0003, 0.0222,
     0.018, 0., 0., 0.],
    [0.0224, 0.0516, 0.0567, 0.0274, 0.0021, 0.0086, 0.0142, 0.0002, 0.0008,
     0.0051, 0., 0.0001, 0.0048],

    [0.0128, 0.0668, 0.049, 0.0584, 0.1764, 0.4577, 0.2761, 0.0059, 0.0232,
     0.0427, 0., 0.0017, 0.0212],
    [0.0317, 0.0121, 0.0198, 0.0428, 0.0937, 0.3487, 0.4068, 0.0072, 0.0231,
     0.0369, 0.0009, 0.002, 0.0157],
    [0.033, 0.0144, 0.0198, 0.2618, 0.2906, 0.4432, 0.0386, 0.0087, 0.0257,
     0.0384, 0.001, 0.0018, 0.0198],

    [0.0841, 0.0528, 0.072, 0.0539, 0.0844, 0.0546, 0.0621, 0.0957, 0.1871,
     0.1575, 0.0094, 0.0139, 0.0418],
    [0.0705, 0.1211, 0.0444, 0.0165, 0.0315, 0.0225, 0.0183, 0.0846, 0.3574,
     0.2594, 0.0029, 0.0102, 0.0212],
    [0.0998, 0.0072, 0.0089, 0.2618, 0.0343, 0.0225, 0.0209, 0.0587, 0.1182,
     0.0427, 0.0038, 0.0124, 0.0262],

    [0., 0.0018, 0.0028, 0.0068, 0.0297, 0.0125, 0.0084, 0.0381, 0.017, 0.0128,
     0.021, 0.3249, 0.3014],
    [0.0025, 0.0001, 0.0003, 0.002, 0.0045, 0.0016, 0.0004, 0.0149, 0., 0.0031,
     0.1865, 0.3535, 0.2968],
    [0.0021, 0., 0.0002, 0.2618, 0.0004, 0.0014, 0.0003, 0.0141, 0., 0.0019,
     0.1062, 0.3321, 0.0379]])

target_corr = np.array([[0.123, 0.145, 0.112, 0.113],
                        [0.145, 0.197, 0.163, 0.193],
                        [0.112, 0.163, 0.211, 0.058],
                        [0.113, 0.193, 0.058, 0.186]])

workingdir = os.getcwd()


def create_individual():
    return origin_map


# def create_individual():
#     new_map = origin_map + np.multiply(0.1 * origin_map,
#                                        np.random.randn(13, 13))
#     return new_map


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


def cxOnePointStr(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint = randint(1, size - 1)
    indCls = type(ind1)

    new1, new2 = ind1[:cxpoint], ind2[:cxpoint]
    new1 = np.concatenate((new1, ind2[cxpoint:]))
    new2 = np.concatenate((new2, ind1[cxpoint:]))

    return indCls(new1), indCls(new2)


def mutSNP1(ind, p):
    '''
    A bit more sophisticated version: mutation only shifts a character by +/-1.
    '''
    assert (0 <= p <= 1)
    new = ind
    for i, row in enumerate(new):
        for j, item in enumerate(row):
            if np.random.random() <= p:
                if i < 4 and j < 4:  # L2/3
                    mut_sd = mut_degrees[0]
                else:  # other layers
                    mut_sd = mut_degrees[1]
                new[i][j] = item + mut_sd * item * (np.random.randn())
    return type(ind)(new)


def do_and_check(survivors, g):
    if on_server:
        # divide into groups of 5
        for i in range(int(len(survivors) / 5)):
            # do the simulations
            map_ids = np.arange(i * 5, (i + 1) * 5)
            batch_genetic(survivors[i * 5:(i + 1) * 5], g, map_ids)
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
better_inds_evolved = np.zeros((max_generations, 5))
# Target: fitness of pre- vs. post-learning exp. data is 0.036
# which means average difference for individual connection is 0.06
# so evolved fitness should be at least smaller than this level
while min(fits) > 0.025 and g < max_generations:
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

    sorted_inds = np.arange(0, 20)[np.argsort(fits)]
    # save evolved better individuals
    better_inds_evolved[g, :] = sorted_inds[:5]
    np.save(
        workingdir + '/output/inds_evolved_g{:02d}.npy'.format(g),
        better_inds_evolved)

    # delete .gdf files to save space
    worse_inds = sorted_inds[5:]
    for i in worse_inds:
        data_dir = os.getcwd() + '/output/g={0:02d}_ind={1:02d}/'.format(g, i) + 'data/'
        for item in os.listdir(data_dir):
            if item.endswith('.gdf'):
                os.remove(data_dir + item)

    g += 1

plt.figure()
plt.plot(np.arange(g), fitness_evolved[:g, :], 'b.')
plt.hlines(0, 0, g + 10, 'k', linestyles='--')
plt.xlabel('Number of generations')
plt.ylabel('Fitness to minimize')
plt.title("Evolution of 5 best individuals' fitness")
plt.tight_layout()
plt.savefig(workingdir + '/output/genetic_hj.png')
