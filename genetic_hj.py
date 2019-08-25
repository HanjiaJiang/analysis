'''
In this script we start with a population of random sequences of letters and
punctuation marks of a length len('Hello world!') and evolve towards the
friendly greeting.

Idea taken from:
https://www.electricmonk.nl/log/2011/09/28/evolutionary-algorithm-evolving-hello-world/
'''

import string
import numpy as np
from random import choice, randint
from deap import base, creator, tools
import matplotlib.pyplot as plt

target = 'Hello, world!'
N_ind = 20
p_cx = 0.8
p_mut = 0.1
max_generations = 1000
# this will be used as a global variable in a few functions
chars = string.ascii_letters + string.punctuation + ' '

def create_individual(target='Hello world!'):
    new_str = ''.join(choice(chars) for i in range(len(target)))
    return new_str


def evaluate(individual, target='Hello world!'):
    '''
    This function assumes that individual and target are of the same length!
    '''
    fitness = 0
    for t, i in zip(target, individual):
        dif = (ord(t) - ord(i))**2
        fitness += dif
    return (fitness,)

def cxOnePointStr(ind1, ind2):
    '''
    Provides the same functionality as tools.cxOnePoint(), but for strings.
    Creates new individuals based on the given ones
    (strings cannot be modified in place)
    '''
    size = min(len(ind1), len(ind2))
    cxpoint = randint(1, size - 1)
    indCls = type(ind1)

    new1, new2 = ind1[:cxpoint], ind2[:cxpoint]
    new1 += ''.join(ind2[cxpoint:])
    new2 += ''.join(ind1[cxpoint:])

    return indCls(new1), indCls(new2)

def mutSNP(ind, p):
    '''
    Mimics single nucleotide polymorphism, but applied to all elements of ind.
    '''
    assert(0 <= p <= 1)
    new = list(ind)
    for i in range(len(new)):
        if np.random.random() <= p:
            new[i] = choice(chars)
            #print(new[i]+'\t', end='')
    #print('\n')
    new = ''.join(new)
    return type(ind)(new)

def mutSNP1(ind, p):
    '''
    A bit more sophisticated version: mutation only shifts a character by +/-1.
    '''
    assert(0 <= p <= 1)
    new = list(ind)
    for i in range(len(new)):
        if np.random.random() <= p:
            new_char = chr(ord(new[i]) + randint(-1, 1))
            if new_char not in chars:
                new_char = choice(chars)
            new[i] = new_char
            #print(new[i]+'\t', end='')
    #print('\n')
    new = ''.join(new)
    return type(ind)(new)



# fitness function should minimize the difference between present and target str
creator.create('FitMin', base.Fitness, weights=(-1, ))
# individual is a string of a correct length
creator.create('Individual', str, fitness=creator.FitMin) #, n=len(target))

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
for ind in population:
    ind.fitness.values = box.evaluate(ind)

### EVOLUTION
fits = [i.fitness.values[0] for i in population]
for i in np.argsort(fits):
    print(population[i], population[i].fitness.values)

g = 0
fitness_evolved = np.zeros((max_generations, 5))
while min(fits) > 0 and g < max_generations:
    print('generation', g, 'FitMin', min(fits))

    ## SELECTION
    survivors = box.select(population, len(population))
    # since all our functions for genetic modifications create new individuals
    # instead of changing them in place, we do not need to clone the survivors

    ## GENETIC OPERATIONS
    # crossing-over
    half = int(len(survivors)/2)
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

    # EVALUATION
    for ind in survivors:
        ind.fitness.values = box.evaluate(ind)

    population[:] = survivors
    fits = [i.fitness.values[0] for i in population]

    #print([(population[i], population[i].fitness.values) for i in np.argsort(fits)[:5]])
    fitness_evolved[g,:] = np.array([population[i].fitness.values for i in np.argsort(fits)[:5]]).reshape(5)
    g += 1

print(g, min(fits))
for i in np.argsort(fits):
    print(population[i], population[i].fitness.values)

plt.figure()
plt.plot(np.arange(g), fitness_evolved[:g,:])
plt.hlines(0, 0, g+10, 'k', linestyles='--')
plt.xlabel('Number of generations')
plt.ylabel('Fitness to minimize')
plt.title("Evolution of 5 best individuals' fitness")
plt.tight_layout()
plt.show()
