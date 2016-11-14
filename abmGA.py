import camerasrandomised as cam
import numpy as np 
import random
import functools
from operator import add
from multiprocessing import Pool
import time 
# create target counts
'''tobeaveraged = [cam.runProgram(cam.reshaped, cam.length, cam.width) for i in range (0,30)]
basearray = np.array(tobeaveraged[0])
for i in range(1, 30):
    basearray = basearray + np.array(tobeaveraged[i])    
avgcounts = basearray/len(tobeaveraged)

basecounts = avgcounts '''

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
    
def grade(pop, target):
    '''Determines average fitness'''
    summed = functools.reduce(add, (fitness(x, target) for x in pop), 0)
    return summed / (len(pop)  * 1.0)      
    
def evolve(pop, target, retain=0.1, random_select = 0.01, mutate = 0.02):
    '''Evolve function performs the GA. The retain rate determines how many
    individuals are selected as possible parents. The random select decides
    how many of the rest of the population are randomly selected to also be
    a parent. The mutation rate determines the probability that a child will
    mutate (or more precisely, the probability that an element of the child
    will mutate)'''
    #graded = [ (fitness(x, target), x) for x in pop]
    #graded = p.map(fitness, pop)    
   
    p = Pool()
    graded = [p.map(fitness, pop), pop]
    graded = [ (graded[0][i], graded[1][i]) for i in range(len(graded[0])) ]
    
    grades = [item[0] for item in graded]
    avg_grade = sum(grades) / len(grades)
    global fitness_history
    fitness_history.append(avg_grade)
    
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
    return parents
    
p = population(20)
target = basecounts
#fitness_history = [grade(p, target),]
fitness_history = []
 
#%%

t0 = time.time()

for i in range(2):
    if __name__ == '__main__':
        p = evolve(p, target)
    print(p)
    print(i)
    #fitness_history.append(grade(p, target)) '''
    
t1 = time.time()

total = t1 - t0

print(total)    
    
'''
pop = population(50)
    

def run():
    print('you have entered the function')
    print(__name__)
    print('you have entered the if clause')
    print(__name__)
    p = Pool(2)
    print('you have initiated the pool')
    t0 = time.time()
    graded = [p.map(fitness, pop), pop]
    print('you have made it!')
    graded = [ (graded[0][i], graded[1][i]) for i in range(len(graded[0])) ]
    t1 = time.time()

    total = t1 - t0
    print(total)
    
        
if __name__ == '__main__':
    run()   '''
    
    
    
#for datum in fitness_history:
 #   print(datum)
    

    
    
    
#%%
    
'''import cameras
import numpy as np
import random
import functools
from operator import add'''

'''basecounts = cameras.runProgram(cameras.reshaped, cameras.length, cameras.width)

#%%


length = 10
width = 10
minimum = 0
maximum = 4

def individual():
    'Create a member of the population.'
    minimum = 0
    maximum = 4
    mrand = np.random.randint(minimum, maximum, size = (length,width))
    np.fill_diagonal(mrand, 0)
    reshaped = list(mrand.flatten())
    return reshaped
  
    
def population(count):
    
    return [individual() for x in range(count)]

def fitness(individual, target):
    #counts = cameras.runProgram(individual, length, width)
    individualcounts = cameras.runProgram(individual, length, width)
    fitness = 0
    for i in range(0,5):
        fitness = fitness + [x == y for (x, y) in zip(individualcounts[i], target[i])].count(True)
    return fitness
    ##code to compare counts with basecounts
    
    
def grade(pop, target):
    
    summed = functools.reduce(add, (fitness(x, target) for x in pop), 0)
    return summed / (len(pop)  * 1.0)      
    
def evolve(pop, target, retain=0.1, random_select = 0.05, mutate = 0.5):
    graded = [ (fitness(x, target), x) for x in pop]
    graded = sorted(graded)
    graded.reverse()
    graded = [ x[1] for x in graded]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    print('Parents have been found!')
    
    # randomly add other individuals
    for individual in graded[retain_length:]:
        if random_select > random.random():
            parents.append(individual)
            print('Randomers have been addded')
    
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
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
            print('A child has been made')
            
    for individual in children:
        if mutate > random.random():
            print('A mutation has happened!')
            pos_to_mutate = random.randint(0, len(individual)-1)
            if pos_to_mutate%11 == 0:
                pass
            else:
                individual[pos_to_mutate] = random.randint(minimum, maximum)
            
    parents.extend(children)
    return parents
    
p = population(10)
target = basecounts
fitness_history = [grade(p, target),]

    
for i in range(10):
    print('We have entered the loop')
    p = evolve(p, target)
    print(p)
    fitness_history.append(grade(p, target))
    
    
for datum in fitness_history:
    print(datum) '''