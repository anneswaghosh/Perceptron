import numpy as np
import math

def file_len (filename):
  len = 0
  train_f = open (filename, "r")
  for line in train_f:
    len += 1

  return len

def create_table (filename):
  num_rows = file_len (filename)
  num_cols = 20
  i = 0

  table = np.zeros ((num_rows, num_cols))

  train_f = open (filename, "r")
  for line in train_f:
    s = line.split (' ')
    
    table[i][0] = int (s[0])

    for x in range (1, len(s)):
      s_split = s[x].split (':')
      table[i][int (s_split[0])] = float (s_split[1])

    i += 1

  return table

def perceptron (train_table, W, bias, rate):
  splitted = np.split (train_table, [1], axis = 1)

  Y = splitted[0]
  X = splitted[1]

  for eg in range (X.shape[0]):
    if (((np.dot (W, X[eg]) + bias) * Y[eg]) < 0):
      W = W + (rate * X[eg] * Y[eg])
      bias = bias + (rate * Y[eg])

  return W


train_file = "dataset/diabetes.train"

train_table = create_table (train_file)

weight = np.random.uniform (-0.01, 0.01, train_table.shape[1] - 1)
bias = np.random.uniform (-0.01, 0.01)
rate = 0.01

weight_vec = perceptron (train_table, weight, bias, rate)

  
test_file = "dataset/diabetes.test"

test_table = create_table (test_file)
