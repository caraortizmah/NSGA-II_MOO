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

#global pcc_d

sys.setrecursionlimit(100000)

#****
#Initialization of matrices

global pcc_d
global cont_pcc
global cont_unif

def init():
  """
  Initialization of matrices that depends on the sliding_window function
  """
  global array_r
  global train_scores
  global exp_scores
  global pred_scores
  global tr_sc
  global train_char
  global num_scores
  global pr_sc

  print("*Initialization of the matrices")

  array_r = np.loadtxt('matrix_DR1_label.txt', dtype=str) #matrix data by position and amino acid
  #train_scores = np.loadtxt('train1_DRB1_0101_e.txt', dtype=str)
  train_scores = np.loadtxt('train_setE.txt', dtype=str)
  exp_scores=np.ones((train_scores.shape[0]), dtype='f')
  pred_scores=np.ones((train_scores.shape[0]), dtype='f')

  print("*Adding hashable objects in the peptide data")

  tr_sc=sliding_window(train_scores)

  print("*Expanding peptide data by sliding window")

  train_char=np.chararray((tr_sc.shape[0],9))#9
  pr_sc=np.ones((tr_sc.shape[0]), dtype='f')
  num_scores=np.zeros(train_char.shape, dtype='f')

  secondary_m()

  array_r=reorganizing_labels(array_r, tr_sc[0:tr_sc.shape[0],0])

def secondary_m():
  """
  """
  for i in range(train_scores.shape[0]): #number of train_scores' rows of the original set
    exp_scores[i]=train_scores[i,1] #extracting scores of the train_scores vector

  rows=tr_sc.shape[0]#number of rows train_scores   
  i=0
  for a in tr_sc[0:rows,0]: #1selecting the peptides column
    #print (a)
    train_char[i,:]=[b for b in a[0:9]]#9
    i+=1  

def reorganizing_labels(arr, tr_sc_a):
  """
  """
  pep=np.array(('A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'))
  ar = np.zeros((20,9), dtype='d')
  a = np.zeros((20), dtype='d')
  for i in tr_sc_a:
    l=0
    for j in i:
      for k in range(20):
        if j==pep[k]:
          ar[k,l]+=1
      l+=1
  for i in range(20):
    a[i]=np.sum(ar[i,0:9])
  
  dict={}
  for i in range(20):
    dict[pep[i]]=a[i]

  bb = sorted(dict.items(), key=operator.itemgetter(1))
  bb.reverse()

  for i in range(20):
    for j in range(20):
      if bb[i][0] == arr[0,0:20][j]:
        aux=np.copy(arr[0:10,j])
        arr[0:10,j]=arr[0:10,i]
        arr[0:10,i]=aux

  return arr

def hashing_train(nump_arr):
  """
  This function creates a list of hashable from peptides of the column 0 of the 
  nump_arr (numpyarray).
  After that, function stacks nump_arr with the list of hashable data in a new
  numpy array as dtype=object.
  The lines below are the resume of the command in the return

  ha = [i.__hash__ for i in nump_arr[0:93,0]]
  hh=np.array(ha)
  y= np.array(nump_arr, dtype=object)
  xx=np.column_stack((y,hh))
  """
  col=nump_arr.shape[0]
  return np.column_stack((np.array(nump_arr, dtype=object),np.array([i.__hash__ for i in nump_arr[0:col,0]])))

def sliding_window(train):
  """
  This function returns the first 9 characters of the string (having more than 9 characters) until the last 9 characters of that string.
  At the end, the original string is deleted. All elements in the same row of the original string are copied to the new shorter 9-character strings.
  From a string vector:

  ABCDEFGHIJKL 0.05
  XXXYYYZZZ    0.30
  BBCCAADDFGH  0.85

  The sliding_window() returns:

  ABCDEFGHI  0.05
  BCDEFGHIJ  0.05
  CDEFGHIJK  0.05
  DEFGHIJKL  0.05
  XXXYYYZZZ  0.30
  BBCCAADDF  0.85
  BCCAADDFG  0.85
  CCAADDFGH  0.85

  """
  tr_sc_ha=hashing_train(train) #tr_sc_ha=train_score_hash

  tr_sc_ha_n=tr_sc_ha
  index=index_n=0
  for i in tr_sc_ha:
    if (len(i[0]) > 9 ):
      for j in range(1+len(i[0])-9):
        index_n+=1
        tr_sc_ha_n = np.insert(tr_sc_ha_n,index+j+1,np.array((i[0][j:j+9],i[1],i[2])),0)
      tr_sc_ha_n = np.delete(tr_sc_ha_n, index, axis=0)
    else:
      index_n+=1
    index=index_n
  return tr_sc_ha_n

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
    return [random.uniform(a, b) for a, b in zip([-0.0001] * size, [-0.0001] * size)]
  try:
    return [random.uniform(a, b) for a, b in zip(low, up)]
  except TypeError:
    return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def scores_assgn_old():
  """
  """
    
  i=j=k=0
  for i in range(train_char.shape[0]): # i is the number of data (rows)
    for j in range(train_char.shape[1]): # j is the number of pockets (columns)
      num_sc=assign_aa_value(i, j, k)

  return num_sc

def assign_aa_value(i, j, k):
  """
  assign_aa_value is a recursive function aiming to find amino acid coincidence from the tt train matrix with the aa matrix and put its value in the n_sc matrix.
  For getting better readability assing_aa_value does the following code using a loop for:

  for k in range(aa.shape[1]): # k is the number of total of amino acids
    if tt[i,j].decode('UTF-8')==aa[0,k]: #finding coincidence of the residue
      n_sc[i,j]=aa[j+1,k] #assigning the score's coincidence
      break
  """
  global array_r #aa
  global train_char #tt
  global num_scores #n_sc

  if k<array_r.shape[1]:
    if train_char[i,j].decode('UTF-8')==array_r[0,k]:
      num_scores[i,j]=array_r[j+1,k]
    return assign_aa_value(i, j, k+1)
  else:
    return num_scores

def scores_assgn():
  """
  """

  i=j=k=0
  for i in range(train_char.shape[0]): # i is the number of data (rows)
    for j in range(train_char.shape[1]): # j is the number of pockets (columns)
      for k in range(array_r.shape[1]): #finding coincidence of the residue
        if train_char[i,j].decode('UTF-8')==array_r[0,k]: #finding coincidence of the residue
          num_scores[i,j]=array_r[j+1,k] #assigning the score's coincidence

  return num_scores

def stacking_old(i):
  """
  Recursive function for selecting the bigger score among rows with the same hash
  and deleting the others as repetitions.
  The new array has rows without repeated hash
  """

  global cont

  if i<cont.shape[0]-1:
    if cont[i,1]==cont[i+1,1]:
      if cont[i,0]>cont[i+1,0]:
        cont=np.delete(cont, i+1, axis=0)
      else:
        cont=np.delete(cont, i, axis=0)
      return stacking(i)
    else:
      return stacking(i+1)
  else:
    return np.array(cont[0:cont.shape[0],0], dtype='float32')

def stacking():
  """
  Function for selecting the bigger score among rows with the same hash
  and deleting the others as repetitions.
  The new array has rows without repeated hash
  """
  global cont
  i=0

  while i<cont.shape[0]-1:
    if cont[i,1]==cont[i+1,1]:
      if cont[i,0]>cont[i+1,0]:
        cont=np.delete(cont, i+1, axis=0)
      else:
        cont=np.delete(cont, i, axis=0)
    else:
      i+=1
  return np.array(cont[0:cont.shape[0],0], dtype='float32')

def sum_weights_old(a,b,pr):
  """
  This function sums all elements that come from a[i,:]*b[:] and saves them in pr matrix.
  It is defined for a specific task for num_scores and indv arrays.
  Multiplication array has the same lenght.

  """
  #pr[i]=np.sum(a[i,:]*b[:])
  i=0
  #print ("individuals: ", b[:])
  for i in range(tr_sc.shape[0]): #sum of all residue scores of the peptide train_scores
    #print("num_scores: ", num_scores[i,:])
    sum_s=a[i,:]*b[:]#weight
    pr[i]=np.sum(sum_s)#sum all elements of sum_s array
  return pr

def sum_weights(a,b):
  """
  This function sums all elements that come from a[i,:]*b[:] and saves them in pr matrix.
  It is defined for a specific task for num_scores and indv arrays.
  Multiplication array has the same lenght.

  """
  return a.dot(b)

def corr_pcc(indv):

  global pred_scores
  global pr_sc
  global cont
  global cont_pcc

  cont_pcc+=1

  num_scores=scores_assgn()
  #pr_sc=sum_weights(num_scores,indv)#,pr_sc)
  pr_sc=num_scores.dot(np.array(indv))
  cont=np.column_stack((pr_sc,tr_sc[0:tr_sc.shape[0],2]))
  
  pred_scores=stacking()
  #print ("exp: ",exp_scores, " pred: ", pred_scores)
  pcc = pearsonr(exp_scores,pred_scores)
  pcc_d = pcc[0]
  #print (" ",cont_pcc, "pcc= ", pcc_d)
  #if cont_pcc == 20: 
  #  cont_pcc = 0
  #  print(" ")
  return pcc[0], pcc[0]

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

    init()
    cont_pcc = 0
    cont_unif = 0
    pop, stats = main(int(sys.argv[1]), int(sys.argv[2]))
    j=0
    #print("pop size: ", len(pop))
    #for i in pop:
    #    j+=1
    #    print("number: ",j, " ", i, "type: ", type(i))

    #pop.sort(key=lambda x: x.fitness.values)
    
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