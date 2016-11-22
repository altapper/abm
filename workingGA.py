
import numpy as np 
import random

random.seed(100)
np.random.seed(100)
import workingcameras as cam


import functools
from operator import add
import time


basecounts = cam.runProgram(cam.reshaped, cam.length, cam.width)

#%%


length = 10
width = 10
minimum = 0
maximum = 0.4

def individual():
    '''Create a member of the population (a flattened origin-destination matrix) '''
    #mrand = 0.04 * np.random.random_sample((length,width))
    mrand = np.random.randint(0, 4, (10,10))
    np.fill_diagonal(mrand, 0)
    reshaped = list(mrand.flatten())
    return reshaped
  
    
def population(count):
    '''Creates a population of members'''
    return [individual() for x in range(count)]

def fitness(individual):
    '''Determines the fitness of an individual
    First calculates the camera counts for the origin-destination matrix, and
    then compares them to the target counts'''
    
    
    '''tobeaveraged = [cam.runProgram(individual, length, width) for i in range (0,30)]
    basearray = np.array(tobeaveraged[0])
    for i in range(1, 30):
        basearray = basearray + np.array(tobeaveraged[i])    
    avgcounts = basearray/len(tobeaveraged) '''
    
    counts = cam.runProgram(individual, length, width)
    fitness = 0
    for i in range(0, 5):    
        fitness = fitness + sum([abs(x - y) for (x, y) in zip(counts[i], basecounts[i])])
        print(fitness)
    return fitness
    ## SMALLER IS BETTER
   
#%%

from multiprocessing import Pool

   
#%%    
  #
print(__name__)   

fitness_history = []

def evolve(pop, retain = 0.2, random_select = 0.1, mutate = 0.1):
    print('you have entered the function')
    p = Pool()
    print('you have initiated the pool')
    t0 = time.time()
    graded = [p.map(fitness, pop), pop]
    print('you have made it!')
    p.close()
    print('you have closed the pool')
    graded = [ (graded[0][i], graded[1][i]) for i in range(len(graded[0])) ]

    grades = [item[0] for item in graded]    
    avg_grade = sum(grades) / len(grades)
    global fitness_history
    fitness_history.append(avg_grade)
    
    test = [x for x in sorted(graded)]
    global best
    best.append(test[0])
        
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    
    
    # randomly add other individuals
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)
    
    '''The crossover is performed as a simple half/half splicing'''
    parents_length = len(parents)
    print(parents_length)
    print(len(parents))
    desired_length = len(pop) - parents_length
    print(desired_length)
    children = []
    while len(children) < desired_length:
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) // 2
            child = male[:half] + female[half:]
            children.append(child)
            
    for individual in children:
        for pos_to_mutate in range(0, len(individual)):
            if pos_to_mutate%11 == 0:
                pass
            elif mutate > random.random():
                print('A mutation has happened!')
                #individual[pos_to_mutate] = 0.04 * np.random.random_sample()
                individual[pos_to_mutate] = random.randint(0,4)
            
            
            
            
    parents.extend(children)
    
    t1 = time.time()

    total = t1 - t0
    print(total)
    return parents

p = []    
localmin = []
best = []

for j in range(3):    
    p = population(20)
    for i in range(5):
        if __name__ == '__main__':
            print('you have entered the if clause')
            p = evolve(p)
    localmin.append(p[0])

        
print(best)
print(localmin[0])
print('well and truly out')


def assess(pop):
    print('you have entered the function')
    p = Pool()
    print('you have initiated the pool')
    graded = [p.map(fitness, pop), pop]
    print('you have made it!')
    p.close()
    print('you have closed the pool')
    graded = [ (graded[0][i], graded[1][i]) for i in range(len(graded[0])) ]    
    return graded


localmingraded = []    
a = []

if __name__ == '__main__':
    print('and back in again')
    a = assess(localmin)
    
localmingraded.append(a)
localmingraded = localmingraded[0]

columnheadings = [item[0] for item in localmingraded]
data = [item[1] for item in localmingraded]

import pandas as pd

def writeGenes(dataIn, columnsIn):
    datatranspose = list(map(list, zip(*dataIn)))
    df = pd.DataFrame(data = datatranspose, columns = columnsIn)
    df.to_csv('genestest.csv')
    pass

writeGenes(data, columnheadings)
