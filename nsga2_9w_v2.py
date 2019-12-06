#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, division, print_function, unicode_literals
import array
import random
import json
import operator

import numpy as np
import ctypes

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools

from scipy.stats import pearsonr
import os
import sys

import corr1
import corr2
import corr3

sys.setrecursionlimit(100000)

global pcc_d
global cont_pcc
global cont_unif
global array_r
global train_char
global num_scores
global index_n
global label
global hasha
global exp_scores


creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_LOW, BOUND_UP = -1.0, 1.0

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 9
#count_1 = 0

def uniform(low, up, size=None):
  global cont_unif

  cont_unif+=1
  if cont_unif<2:
    return [random.uniform(a, b) for a, b in zip([0.05] * size, [0.05] * size)]
  try:
    return [random.uniform(a, b) for a, b in zip(low, up)]
  except TypeError:
    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def corr_pcc(indv):

  global array_r
  global train_char
  global num_scores
  global label
  global hasha
  global index_n
  global exp_scores

  pcc = corr1.corr_nsga2(np.array(indv), label, train_char, hasha, array_r, exp_scores, index_n)

  return pcc, pcc

toolbox.register("evaluate", corr_pcc)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)

def main(NGEN, MU, seed=None):
    
  print("*")
  print("*Starting evaluation")
  print("")
  random.seed(seed)

  #NGEN = 25
  #MU = 20#100
  CXPB = 0.3

  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean, axis=0)
  stats.register("std", np.std, axis=0)
  stats.register("min", np.min, axis=0)
  stats.register("max", np.max, axis=0)
    
  logbook = tools.Logbook()
  #logbook.header = "gen", "evals", "std", "min", "avg", "max"
  #logbook.header = "gen", "evals"#, "min", "max", "individual"

  pop = toolbox.population(n=MU)

  # Evaluate the individuals with an invalid fitness
  invalid_ind = [ind for ind in pop if not ind.fitness.valid]
  fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

  for ind, fit in zip(invalid_ind, fitnesses):
      ind.fitness.values = fit

  # This is just to assign the crowding distance to the individuals
  # no actual selection is done
  pop = toolbox.select(pop, len(pop))
    
  record = stats.compile(pop)
  logbook.record(gen=0, evals=len(invalid_ind), **record)
  print(logbook.stream)

  print("      ")
  print(" Individuals in generation 0: ")
  print("      ")
  j=0
  for i in pop:
    j+=1
    print(j, ":", i, "PCC=",i.fitness)
  print("      ")

  # Begin the generational process
  for gen in range(1, NGEN):
      # Vary the population
      offspring = tools.selTournamentDCD(pop, len(pop))
      offspring = [toolbox.clone(ind) for ind in offspring]
        
      for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
          if random.random() <= CXPB:
              toolbox.mate(ind1, ind2)
            
          toolbox.mutate(ind1)
          toolbox.mutate(ind2)
          del ind1.fitness.values, ind2.fitness.values
        
      # Evaluate the individuals with an invalid fitness
      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
      for ind, fit in zip(invalid_ind, fitnesses):
          ind.fitness.values = fit

      # Select the next generation population
      pop = toolbox.select(pop + offspring, MU)
      record = stats.compile(pop)
      logbook.record(gen=gen, evals=len(invalid_ind), **record)
      print(logbook.stream)

      print("      ")
      print(" Individuals in generation",gen,": ")
      print("      ")
      j=0
      for i in pop:
        j+=1
        print(j, ":", i, "PCC=", i.fitness)
      print("      ")

  #print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))


  return pop, logbook
        
if __name__ == "__main__":


    # with open("pareto_front/zdt1_front.json") as optimal_front_data:
    #     optimal_front = json.load(optimal_front_data)
    # Use 500 of the 1000 points in the json file
    # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
    print("************Genetic Algorithm************")
    print("************Binding Predictor************")
    print("*")
    print("*")
    print("*")
    print("*Number of Generations: ", sys.argv[1])
    print("*Number of Mutations ", sys.argv[2])

    #init()
    cont_pcc = 0
    cont_unif = 0

    array_r = np.zeros((9,20), dtype='f')

    array_r, label, index_n = corr2.store_nsga2()

    train_char = np.array((index_n,9), dtype='U')
    hasha = np.array((index_n), dtype='U')
    exp_scores = np.zeros((8424), dtype='f')

    train_char, hasha, exp_scores = corr3.store_nsga2(index_n) #, hasha, exp_scores

    pop, stats = main(int(sys.argv[1]), int(sys.argv[2]))
    
    print(stats)
    print("--------")
    #print(toolbox.individual())
    #print(toolbox.individual())
    #print("Convergence: ", convergence(pop, optimal_front))
    #print("Diversity: ", diversity(pop, optimal_front[0], optimal_front[-1]))
    
    # import matplotlib.pyplot as plt
    # import numpy
    
    #front = np.array([ind.fitness.values for ind in pop])
    #optimal_front = np.array(optimal_front)
    #plt.scatter(optimal_front[:,0], optimal_front[:,1], c="r")
    #plt.scatter(front[:,0], front[:,1], c="b")
    #plt.axis("tight")
    #plt.show()
