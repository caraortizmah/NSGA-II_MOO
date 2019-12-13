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
  #train_scores = np.loadtxt('train_setE.txt', dtype=str)
  train_scores = np.loadtxt('results_best_weights-GAP1', dtype=str)
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

BOUND_LOW, BOUND_UP = 1.0, 1.0
NDIM = 9

def uniform(low=BOUND_LOW, up=BOUND_UP, size=NDIM):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

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

def stacking_old(cont):
  """
  Function for selecting the bigger score among rows with the same hash
  and deleting the others as repetitions.
  The new array has rows without repeated hash
  """
  #global cont
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

def stacking(cont):
  """
  Function for selecting the bigger score among rows with the same hash
  and deleting the others as repetitions.
  The new array has rows without repeated hash
  """
  #global cont
  i=0

  while i<cont.shape[0]-1:
    if cont[i,3]==cont[i+1,3]:
      if cont[i,0]>cont[i+1,0]:
        cont=np.delete(cont, i+1, axis=0)
      else:
        cont=np.delete(cont, i, axis=0)
    else:
      i+=1
  return np.array(cont)

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
  global info_1
  global tr_sc

  num_scores=scores_assgn()
  pr_sc=num_scores.dot(np.array(indv))
  #cont=np.column_stack((pr_sc,tr_sc[0:tr_sc.shape[0],2]))

  cont=np.column_stack((pr_sc,tr_sc))
  #print(":",pr_sc[0], tr_sc[0])
  #print("stacking: ", cont.shape)
  #print("stacking: ", cont[0])
  info=stacking(cont)
  #print("info: ", info.shape)
  info_1=np.column_stack((exp_scores,info))

  pred_scores=np.array(info[:,0], dtype='float32')

  pcc = pearsonr(exp_scores,pred_scores)
  pcc_d = pcc[0]

  return pcc[0], pcc[0]

def roc():

  global pred_scores
  global exp_scores

  resol = 150

  tpr = np.zeros((150), dtype='f')
  fpr = np.zeros((150), dtype='f')

  max_v = 0.0
  for i in pred_scores:
    if max_v < i: 
      max_v = i
  #print(max_v)
  min_v = max_v
  for i in pred_scores:
    if min_v > i:
      min_v = i
  #print(min_v)
  
  threshold_exp = 0.426

  diff_max = max_v - min_v
  diff_min = diff_max / resol
  threshold_pred = min_v
  #print(diff_max, diff_min)
  print(" TPR      FPR")
  for i in range(resol):
    
    threshold_pred = min_v + (diff_min * i)
    tp = fn = tn = fp = 0.0 
    #true positives (tp), false negatives (fn), true negatives (tn), false positives (fp)

    for j in range(len(exp_scores)):
      if exp_scores[j]>threshold_exp: #below of the experimental threshold is negative
        if pred_scores[j]>threshold_pred: #negative for predicted
          tn = tn + 1
        else: #positive for predicted
          fp = fp + 1
      else: #above or equal of the experimental threshold is positive
        if pred_scores[j]>threshold_pred: #negative for predicted
          fn = fn + 1
        else: #positive for predicted
          tp = tp + 1
    tpr[i] = tp / (tp + fn)
    fpr[i] = fp / (tn + fp)
    #print(tp, fn, fp, tn, tpr[i], fpr[i], threshold_pred)
    #print(tpr[i], fpr[i])
    #print("% 1.8f  %1.8f  " %(tpr[i], fpr[i]))
    #print(tpr[i], fpr[i], threshold_pred)
  
  auc = 0.0
  diff_1 = 0.0
  diff_2 = 0.0
  for i in range(resol):
    auc = (fpr[i] - diff_1)*((tpr[i] + diff_2) / 2) + auc #trapezoid area
    #auc = (fpr[i] - diff_1)*tpr[i] + auc #rectangle area
    diff_1 = fpr[i]
    diff_2 = tpr[i]

  return auc
        
if __name__ == "__main__":

  global exp_scores
  global pred_scores
  global cont
  global train_char
  global tr_sc

  # with open("pareto_front/zdt1_front.json") as optimal_front_data:
  #     optimal_front = json.load(optimal_front_data)
  # Use 500 of the 1000 points in the json file
  # optimal_front = sorted(optimal_front[i] for i in range(0, len(optimal_front), 2))
  print("************Genetic Algorithm************")
  print("************Binding Predictor************")
  print("*")
  print("*")
  print("*")

  init()
  #indv = uniform()
  indv = [0.9370631828605782, -0.07922944534573004, 0.14037449906309526, 0.7391540126630456, -0.3369918520531541, 0.9310277654325743, 0.25878403543304407, 0.1833097867051305, 0.35002708270663757]
  #indv = [1.00, -1.00, -1.00, 1.00, -1.00, 1.00, 1.00, -1.00, 1.00]
  print("Values of the weights:")
  print(indv)
  correl = corr_pcc(indv)
  print("**")
  print("Experimental binding (Eb), Predicted binding (Pb), core")
  print("**")
  print("       Eb                Pb             core")
  #print(info_1[:,0:4])
  for i in info_1:
    print("% 1.14f  %1.14f   %3s " %(i[0], i[1], i[2]))  
  print("PCC: ", correl[0])
  auc = roc()
  print("AUC: ", auc)
  print("--------")